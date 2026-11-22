//! STL-10 image recognition dataset as a device-resident
//! [`oa::data::Dataset`].
//!
//! `stl10_binary.tar.gz` is materialized through
//! [`ensure_pack`](crate::pack::ensure_pack) under `stl10/1.0.0`; the
//! `stl10_binary/*_X.bin` members decode from column-major `3 x 96 x 96`
//! u8 records into a `[N, 3, 96, 96]` F32 NCHW image matrix and `*_y.bin`
//! labels `1..=10` into `[N]` F32 class indices `0..=9` into [`CLASSES`].
//! [`oa::data::Dataset::get_item`] returns the `[3, 96, 96]` CHW slice so
//! [`oa::data::collate`] produces `[B, 3, 96, 96]` batches.

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::decode::{Decoded, MatrixData};
use crate::extract;
use crate::pack::{self, FetchProgress};

#[path = "stl10.gen.rs"]
mod generated;

pub(crate) use generated::STL10_PACK;
pub use generated::{CLASSES, IMAGE_SIDE, Stl10Config, Stl10Split};

const PIXELS: usize = IMAGE_SIDE * IMAGE_SIDE;
const IMAGE_BYTES: usize = 3 * PIXELS;

/// Device-resident STL-10 split implementing [`oa::data::Dataset`].
///
/// `images()` is `[N, 3, 96, 96]` F32 NCHW and `labels()` is `[N]` F32 class
/// indices `0..=9` into [`CLASSES`].
pub struct Stl10 {
	dataset: MatrixDataset,
	split: Stl10Split,
}

impl Stl10 {
	/// Fetch, verify, and decode the pack on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, archive decode errors, and an
	/// error when a binary member's length is not a whole number of
	/// records.
	pub fn decode(config: &Stl10Config) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &Stl10Config,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&STL10_PACK, &root, progress)?;
		let gzip = std::fs::read(dir.join("stl10_binary.tar.gz"))
			.map_err(|error| Error::io("read stl10 archive", error))?;
		let tar = extract::gunzip_bytes(&gzip)?;

		let tag = match config.split {
			Stl10Split::Train => "train",
			Stl10Split::Test => "test",
		};
		let x = extract::tar_member(&tar, &format!("stl10_binary/{tag}_X.bin"))?;
		let y = extract::tar_member(&tar, &format!("stl10_binary/{tag}_y.bin"))?;
		if x.len() % IMAGE_BYTES != 0 || y.len() != x.len() / IMAGE_BYTES {
			return Err(Error::data_loss(format!(
				"stl10 {tag} members disagree: {} image bytes for {} labels",
				x.len(),
				y.len()
			)));
		}

		// Each record is column-major [channel][column][row]; transpose into
		// row-major NCHW while scaling u8 -> f32 by 1/255.
		let count = y.len();
		let scale = if config.normalize { 1.0 / 255.0 } else { 1.0 };
		let mut images = Vec::with_capacity(count * IMAGE_BYTES);
		let mut labels = Vec::with_capacity(count);
		for (index, &label) in y.iter().enumerate() {
			if !(1..=10).contains(&label) {
				return Err(Error::data_loss(format!(
					"stl10 label {label} outside 1..=10"
				)));
			}
			labels.push(f32::from(label - 1));
			let record = &x[index * IMAGE_BYTES..(index + 1) * IMAGE_BYTES];
			for channel in 0..3 {
				for row in 0..IMAGE_SIDE {
					for col in 0..IMAGE_SIDE {
						images.push(f32::from(record[channel * PIXELS + col * IMAGE_SIDE + row]) * scale);
					}
				}
			}
		}
		Ok(Decoded {
			matrices: vec![
				MatrixData {
					name: "images",
					shape: vec![count, 3, 96, 96],
					values: images,
				},
				MatrixData {
					name: "labels",
					shape: vec![count],
					values: labels,
				},
			],
			text: None,
			seq_len: 0,
		})
	}

	/// [`load_on`](Self::load_on) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`load_on`](Self::load_on), plus `Cancelled` when
	/// the callback aborts a fetch.
	pub fn load_with_progress_on(
		engine: &Engine,
		config: &Stl10Config,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Ok(Self {
			dataset: MatrixDataset::new(
				decoded.upload("images", engine)?,
				decoded.upload("labels", engine)?,
			)?,
			split: config.split,
		})
	}

	/// Wrap caller-owned matrices as an STL-10 dataset.
	///
	/// `images` must be rank-4 `[N, 3, 96, 96]` F32 and `labels` rank-1 `[N]`
	/// F32 on the same engine.
	///
	/// # Errors
	///
	/// Returns an error for a non-rank-4 image matrix, plus the count,
	/// dtype, and engine errors of [`MatrixDataset::new`].
	pub fn new(images: Matrix, labels: Matrix, split: Stl10Split) -> Result<Self> {
		if images.shape().len() != 4 {
			return Err(Error::invalid_argument(format!(
				"stl10 images must be rank-4 [N, 3, 96, 96], got {:?}",
				images.shape()
			)));
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

	/// Full `[N]` F32 label matrix on the owning engine; indices into
	/// [`CLASSES`].
	#[must_use]
	pub const fn labels(&self) -> &Matrix {
		self.dataset.y()
	}

	/// Which official split this dataset materializes.
	#[must_use]
	pub const fn split(&self) -> Stl10Split {
		self.split
	}
}

impl Dataset for Stl10 {
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
