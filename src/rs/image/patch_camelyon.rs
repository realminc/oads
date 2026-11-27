//! PatchCamelyon histopathology patch classification.
//!
//! `camelyonpatch_level_2_split_{split}_{x,y}.h5.gz` members are
//! materialized through [`ensure_pack`](crate::core::pack::ensure_pack)
//! under `patch_camelyon/1.0.0`, gunzipped in place to sibling `.h5`
//! files, and decoded through `h5`. Upstream stores
//! `[N, 96, 96, 3]` uint8 HWC patches; decode transposes to `[N, 3, 96,
//! 96]` F32 NCHW (scaled to `[0, 1]` when `config.normalize`) with a `[N]`
//! F32 label vector. `get_item` returns the `[3, 96, 96]` CHW slice.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Error, Result, Runtime};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::h5;
use crate::core::pack::{self, FetchProgress};

#[path = "patch_camelyon.gen.rs"]
mod generated;

pub(crate) use generated::PATCH_CAMELYON_PACK;
pub use generated::{PatchCamelyonConfig, PatchCamelyonSplit};

fn members(split: PatchCamelyonSplit) -> (String, String) {
	let stem = match split {
		PatchCamelyonSplit::Train => "train",
		PatchCamelyonSplit::Validation => "valid",
		PatchCamelyonSplit::Test => "test",
	};
	(
		format!("camelyonpatch_level_2_split_{stem}_x.h5.gz"),
		format!("camelyonpatch_level_2_split_{stem}_y.h5.gz"),
	)
}

/// Gunzip `gz_path` to a sibling `.h5` file when absent (or stale),
/// streaming in fixed chunks. Returns the `.h5` path.
fn gunzip_member(gz_path: &Path) -> Result<PathBuf> {
	let out = gz_path.with_extension("");
	if out.exists() && out.metadata().map(|m| m.len() > 0).unwrap_or(false) {
		return Ok(out);
	}
	let mut source = flate2::read::GzDecoder::new(
		std::fs::File::open(gz_path).map_err(|e| Error::io("open h5.gz member", e))?,
	);
	let mut target = std::io::BufWriter::new(
		std::fs::File::create(&out).map_err(|e| Error::io("create h5 member", e))?,
	);
	let mut chunk = [0u8; 1 << 20];
	loop {
		let read = source
			.read(&mut chunk)
			.map_err(|e| Error::io("gunzip h5 member", e))?;
		if read == 0 {
			break;
		}
		target
			.write_all(&chunk[..read])
			.map_err(|e| Error::io("write h5 member", e))?;
	}
	Ok(out)
}

fn decode_split(
	pack: &str,
	dir: &Path,
	split: PatchCamelyonSplit,
	normalize: bool,
	count: Option<usize>,
) -> Result<Decoded> {
	let (x_name, y_name) = members(split);
	let x_file = gunzip_member(&dir.join(x_name))?;
	let y_file = gunzip_member(&dir.join(y_name))?;
	let x = h5::open_dataset(&x_file, "x")?;
	let y = h5::open_dataset(&y_file, "y")?;
	let rows = count.map_or(u64::MAX, |c| c as u64);
	if x.shape.len() != 4 || x.shape[1..] != [96, 96, 3] || x.element_size != 1 || !x.unsigned {
		return Err(Error::data_loss(format!(
			"{pack} x dataset is not [N, 96, 96, 3] uint8"
		)));
	}
	if !y.unsigned || y.element_size != 1 {
		return Err(Error::data_loss(format!("{pack} y dataset is not uint8")));
	}
	let n = x.shape[0].min(rows);
	if y.shape.first().copied().unwrap_or(0) < n {
		return Err(Error::data_loss(format!(
			"{pack} y dataset shorter than x dataset"
		)));
	}
	let pixels = h5::read_dataset_rows(&x_file, &x, n)?;
	let label_bytes = h5::read_dataset_rows(&y_file, &y, n)?;
	let row = (96 * 96 * 3) as usize;
	let scale = if normalize { 1.0 / 255.0 } else { 1.0 };
	let mut images = Vec::with_capacity(pixels.len());
	for patch in pixels.chunks_exact(row) {
		for channel in 0..3 {
			images.extend((0..96 * 96).map(|i| f32::from(patch[i * 3 + channel]) * scale));
		}
	}
	let n = pixels.len() / row;
	Ok(Decoded {
		matrices: vec![
			MatrixData {
				name: "images",
				shape: vec![n, 3, 96, 96],
				values: images,
			},
			MatrixData {
				name: "labels",
				shape: vec![label_bytes.len()],
				values: label_bytes.iter().map(|&b| f32::from(b)).collect(),
			},
		],
		text: None,
		seq_len: 0,
	})
}

/// Device-resident PatchCamelyon split implementing [`oa::data::Dataset`].
///
/// `images()` is `[N, 3, 96, 96]` F32 NCHW and `labels()` is `[N]` F32
/// class indices `0 | 1`.
pub struct PatchCamelyon {
	dataset: MatrixDataset,
	split: PatchCamelyonSplit,
}

impl PatchCamelyon {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, gunzip errors, and HDF5
	/// decode errors for truncated or unsupported members.
	pub fn decode(config: &PatchCamelyonConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled`
	/// when `progress` aborts a fetch.
	pub fn decode_with_progress(
		config: &PatchCamelyonConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&PATCH_CAMELYON_PACK, &root, progress)?;
		decode_split(
			"patch_camelyon",
			&dir,
			config.split,
			config.normalize,
			config.count,
		)
	}

	/// [`load_on`](Self::load_on) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`load_on`](Self::load_on), plus `Cancelled`
	/// when the callback aborts a fetch.
	pub fn load_with_progress_on(
		engine: &Runtime,
		config: &PatchCamelyonConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Self::new(
			decoded.upload("images", engine)?,
			decoded.upload("labels", engine)?,
			config.split,
		)
	}

	/// Wrap caller-owned matrices as a PatchCamelyon dataset.
	///
	/// `images` must be rank-4 `[N, C, H, W]` F32 and `labels` rank-1 `[N]`
	/// F32 on the same engine.
	///
	/// # Errors
	///
	/// Returns an error for a non-rank-4 image matrix, plus the count,
	/// dtype, and engine errors of [`MatrixDataset::new`].
	pub fn new(images: Matrix, labels: Matrix, split: PatchCamelyonSplit) -> Result<Self> {
		if images.shape().len() != 4 {
			return Err(Error::invalid_argument(
				"patch_camelyon requires [N, C, H, W] images",
			));
		}
		Ok(Self {
			dataset: MatrixDataset::new(images, labels)?,
			split,
		})
	}

	/// Full `[N, 3, 96, 96]` F32 image matrix on the owning engine.
	#[must_use]
	pub const fn images(&self) -> &Matrix {
		self.dataset.x()
	}

	/// Full `[N]` F32 label matrix on the owning engine.
	#[must_use]
	pub const fn labels(&self) -> &Matrix {
		self.dataset.y()
	}

	/// The split this dataset was materialized from.
	#[must_use]
	pub const fn split(&self) -> PatchCamelyonSplit {
		self.split
	}
}

impl Dataset for PatchCamelyon {
	fn len(&self) -> i64 {
		self.dataset.len()
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		self.dataset.get_item(index)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		self.dataset.get_sample(index)
	}
}
