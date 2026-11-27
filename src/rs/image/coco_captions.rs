//! MS COCO 2017 caption pairs as a lazy [`oa::data::Dataset`].
//!
//! `captions_{train,val}2017.json` materialize through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `coco_captions/1.0.0` beside the shared `train2017`/`val2017`
//! image pools. Each of the five caption annotations per image is one
//! sample: `get_item` lazily decodes the `[1, 3, H, W]` F32 NCHW
//! image and `get_sample` pairs it with the caption's `[K]` F32
//! UTF-8 byte slice out of the resident `caption_bytes` blob.
//! `image_ids` carries the source COCO image id per sample.

use std::path::{Path, PathBuf};

use oa::data::{Dataset, Sample};
use oa::image::Image;
use oa::matrix::{self, Matrix};
use oa::{Error, Result, Runtime};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::pack::{self, FetchProgress};
use crate::image::folder;

#[path = "coco_captions.gen.rs"]
mod generated;

pub(crate) use generated::COCO_CAPTIONS_PACK;
pub use generated::{CocoCaptionsConfig, CocoCaptionsSplit};

#[derive(serde::Deserialize)]
struct Captions {
	images: Vec<CocoImage>,
	annotations: Vec<Annotation>,
}

#[derive(serde::Deserialize)]
struct CocoImage {
	id: i64,
	file_name: String,
}

#[derive(serde::Deserialize)]
struct Annotation {
	image_id: i64,
	caption: String,
}

/// Per-caption index: image paths (one row per annotation) plus
/// caption byte spans and COCO image ids.
pub(crate) struct CaptionsIndex {
	pub paths: Vec<PathBuf>,
	pub image_ids: Vec<i64>,
	pub caption_lengths: Vec<u32>,
	pub caption_bytes: Vec<u8>,
}

impl CaptionsIndex {
	fn caption_offsets(&self) -> Vec<i64> {
		let mut offsets = Vec::with_capacity(self.caption_lengths.len() + 1);
		offsets.push(0);
		for &length in &self.caption_lengths {
			offsets.push(offsets.last().copied().unwrap_or(0) + i64::from(length));
		}
		offsets
	}

	fn captions(&self) -> Result<MatrixData> {
		Ok(MatrixData {
			name: "caption_bytes",
			shape: vec![self.caption_bytes.len()],
			values: self
				.caption_bytes
				.iter()
				.map(|&byte| f32::from(byte))
				.collect(),
		})
	}

	fn lengths(&self) -> Result<MatrixData> {
		Ok(MatrixData {
			name: "caption_lengths",
			shape: vec![self.caption_lengths.len()],
			values: self
				.caption_lengths
				.iter()
				.map(|&length| length as f32)
				.collect(),
		})
	}

	fn ids(&self) -> Result<MatrixData> {
		Ok(MatrixData {
			name: "image_ids",
			shape: vec![self.image_ids.len()],
			values: self.image_ids.iter().map(|&id| id as f32).collect(),
		})
	}

	fn paths_text(&self, root: &Path) -> Result<String> {
		let mut text = String::with_capacity(self.paths.len() * 48);
		for path in &self.paths {
			let rel = path
				.strip_prefix(root)
				.map_err(|_| Error::data_loss("coco_captions image path escapes the extracted root"))?;
			text.push_str(&rel.to_string_lossy());
			text.push('\n');
		}
		Ok(text)
	}
}

fn split_key(split: CocoCaptionsSplit) -> &'static str {
	match split {
		CocoCaptionsSplit::Train => "train",
		CocoCaptionsSplit::Validation => "validation",
	}
}

fn index(extracted: &Path, config: &CocoCaptionsConfig) -> Result<CaptionsIndex> {
	let root = extracted.join(split_key(config.split));
	let json = root.join("annotations").join(match config.split {
		CocoCaptionsSplit::Train => "captions_train2017.json",
		CocoCaptionsSplit::Validation => "captions_val2017.json",
	});
	let body = std::fs::read(&json).map_err(|error| Error::io("read coco_captions json", error))?;
	let captions: Captions = serde_json::from_slice(&body)
		.map_err(|error| Error::data_loss(format!("coco_captions json is malformed: {error}")))?;
	let mut names = std::collections::HashMap::with_capacity(captions.images.len());
	for image in captions.images {
		names.insert(image.id, image.file_name);
	}
	let image_root = root.join(match config.split {
		CocoCaptionsSplit::Train => "train2017",
		CocoCaptionsSplit::Validation => "val2017",
	});
	let mut index = CaptionsIndex {
		paths: Vec::with_capacity(captions.annotations.len()),
		image_ids: Vec::with_capacity(captions.annotations.len()),
		caption_lengths: Vec::with_capacity(captions.annotations.len()),
		caption_bytes: Vec::new(),
	};
	for annotation in captions.annotations {
		let name = names.get(&annotation.image_id).ok_or_else(|| {
			Error::data_loss(format!(
				"coco_captions annotation references unknown image {}",
				annotation.image_id
			))
		})?;
		index.paths.push(image_root.join(name));
		index.image_ids.push(annotation.image_id);
		index
			.caption_lengths
			.push(u32::try_from(annotation.caption.len()).unwrap_or(u32::MAX));
		index
			.caption_bytes
			.extend_from_slice(annotation.caption.as_bytes());
	}
	Ok(index)
}

/// Device-resident coco_captions dataset implementing
/// [`oa::data::Dataset`].
///
/// Image members stay in the extracted pack cache; `get_item` lazily
/// decodes one `[3, H, W]` F32 NCHW image at native extent through
/// [`oa::image`]. `caption_bytes()` is the resident flat `[B]` F32
/// UTF-8 blob; `get_sample` slices it to `[K]` per sample.
pub struct CocoCaptions {
	paths: Vec<PathBuf>,
	caption_bytes: Matrix,
	offsets: Vec<i64>,
	image_ids: Vec<i64>,
	split: CocoCaptionsSplit,
	engine: Runtime,
}

impl CocoCaptions {
	/// Decode the selected split and upload it on this thread's
	/// default engine.
	///
	/// # Errors
	///
	/// Fetch, verify, extract, and index the pack on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, archive decode errors,
	/// annotation parse errors, and I/O errors while materializing
	/// the extracted cache.
	pub fn decode(config: &CocoCaptionsConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus
	/// `Cancelled` when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &CocoCaptionsConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let extracted = Self::materialize(config, progress)?;
		let index = index(&extracted, config)?;
		Ok(Decoded {
			matrices: vec![index.captions()?, index.lengths()?, index.ids()?],
			text: Some(index.paths_text(&extracted)?),
			seq_len: 0,
		})
	}

	/// [`load_on`](Self::load_on) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`load_on`](Self::load_on), plus
	/// `Cancelled` when `progress` aborts the fetch.
	pub fn load_with_progress_on(
		engine: &Runtime,
		config: &CocoCaptionsConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		let caption_bytes = decoded.upload("caption_bytes", engine)?;
		let index = index(&Self::extracted_dir(config)?, config)?;
		let offsets = index.caption_offsets();
		Ok(Self {
			paths: index.paths,
			offsets,
			image_ids: index.image_ids,
			caption_bytes,
			split: config.split,
			engine: engine.clone(),
		})
	}

	/// Number of indexed `(image, caption)` samples.
	#[must_use]
	pub fn count(&self) -> usize {
		self.paths.len()
	}

	/// Flat `[B]` F32 caption byte blob on the owning engine;
	/// `caption_lengths`/`offsets` carve one span per sample.
	#[must_use]
	pub const fn caption_bytes(&self) -> &Matrix {
		&self.caption_bytes
	}

	/// COCO image id per sample, in sample order.
	#[must_use]
	pub fn image_ids(&self) -> &[i64] {
		&self.image_ids
	}

	/// Selected split.
	#[must_use]
	pub const fn split(&self) -> CocoCaptionsSplit {
		self.split
	}

	/// Extracted image path for row `index`.
	///
	/// # Errors
	/// Returns `out_of_range` when `index` is outside the dataset.
	pub fn path(&self, index: i64) -> Result<&Path> {
		if index < 0 || index >= self.paths.len() as i64 {
			return Err(Error::out_of_range(
				"coco_captions index is outside the dataset",
			));
		}
		Ok(&self.paths[index as usize])
	}

	/// Lazily decode the image paired with caption `index` to a
	/// `[1, 3, H, W]` F32 NCHW image on the owning engine at native
	/// extent.
	///
	/// # Errors
	/// Returns `out_of_range` for an out-of-range index and the
	/// [`oa::image`] decode errors for the member.
	pub fn image(&self, index: i64) -> Result<Image> {
		folder::decode(&self.engine, self.path(index)?)
	}

	/// `[K]` F32 caption byte slice for sample `index` on the
	/// owning engine, sharing the resident
	/// [`caption_bytes`](Self::caption_bytes) storage.
	///
	/// # Errors
	/// Returns `out_of_range` for an out-of-range index and the
	/// matrix slice error for the span.
	pub fn caption(&self, index: i64) -> Result<Matrix> {
		if index < 0 || index as usize + 1 >= self.offsets.len() {
			return Err(Error::out_of_range(
				"coco_captions index is outside the dataset",
			));
		}
		let start = self.offsets[index as usize];
		let end = self.offsets[index as usize + 1];
		matrix::slice(&self.caption_bytes, 0, start, end)
	}

	fn pack_dir(config: &CocoCaptionsConfig) -> Result<PathBuf> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		Ok(root.join("coco_captions").join("1.0.0"))
	}

	fn extracted_dir(config: &CocoCaptionsConfig) -> Result<PathBuf> {
		Ok(Self::pack_dir(config)?.join("extracted"))
	}

	fn materialize(
		config: &CocoCaptionsConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<PathBuf> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&COCO_CAPTIONS_PACK, &root, progress)?;
		let extracted = dir.join("extracted");
		let split_key = split_key(config.split);
		let done = extracted.join(format!(".done_{split_key}"));
		if done.exists() {
			return Ok(extracted);
		}
		let members: &[&str] = match config.split {
			CocoCaptionsSplit::Train => &["train2017.zip", "annotations_trainval2017.zip"],
			CocoCaptionsSplit::Validation => &["val2017.zip", "annotations_trainval2017.zip"],
		};
		let target = extracted.join(split_key);
		for member in members {
			folder::extract_images(&dir.join(member), &target)?;
		}
		std::fs::write(&done, b"done")
			.map_err(|error| Error::io("write coco_captions extract marker", error))?;
		Ok(extracted)
	}
}

impl Dataset for CocoCaptions {
	fn len(&self) -> i64 {
		i64::try_from(self.paths.len()).unwrap_or(i64::MAX)
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		self.image(index)?.into_matrix().squeeze(0)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		if index < 0 || index >= self.paths.len() as i64 {
			return Err(Error::out_of_range(
				"coco_captions index is outside the dataset",
			));
		}
		let x = self.get_item(index)?;
		let y = self.caption(index)?;
		Ok(Sample::new_with_label(x, y))
	}
}
