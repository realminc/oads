//! EMNIST handwritten letters and digits as a device-resident
//! [`oa::data::Dataset`].
//!
//! The single upstream zip carries all six published variants as IDX gzip
//! members under `gzip/`; [`ensure_pack`](crate::core::pack::ensure_pack)
//! materializes it under `emnist/1.0.0`. The selected variant's split members
//! are inflated, IDX-decoded on the host, and uploaded once as a
//! `[N, 1, 28, 28]` F32 image matrix and a `[N]` F32 label matrix.
//! `get_item` returns the `[1, 28, 28]` CHW slice so [`oa::data::collate`]
//! produces `[B, 1, 28, 28]` NCHW batches.

use std::collections::HashMap;

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::extract;
use crate::core::idx::{self, IdxSplit};
use crate::core::pack::{self, FetchProgress};

#[path = "emnist.gen.rs"]
mod generated;

pub(crate) use generated::EMNIST_PACK;
pub use generated::{EmnistConfig, EmnistSplit};

/// Published EMNIST class groupings inside the shared upstream zip.
///
/// Label values are the raw IDX class indices of each variant:
/// `Balanced`/`ByMerge` `0..=46`, `ByClass` `0..=61`, `Digits`/`Mnist`
/// `0..=9`, and `Letters` `1..=26`. The `emnist-<variant>-mapping.txt`
/// member beside the IDX files maps indices to ASCII codes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EmnistVariant {
	/// 47 balanced classes — digits plus the most frequent letters.
	#[default]
	Balanced,
	/// 62 classes — digits plus uppercase and lowercase letters.
	ByClass,
	/// 47 merged classes — lowercase letters merged into uppercase twins.
	ByMerge,
	/// 10 digit classes.
	Digits,
	/// 26 letter classes (labels are `1..=26`).
	Letters,
	/// 10 classes in MNIST layout — direct drop-in for the digit task.
	Mnist,
}

impl EmnistVariant {
	/// IDX member prefix inside the zip's `gzip/` directory.
	pub const fn member(self) -> &'static str {
		match self {
			Self::Balanced => "gzip/emnist-balanced",
			Self::ByClass => "gzip/emnist-byclass",
			Self::ByMerge => "gzip/emnist-bymerge",
			Self::Digits => "gzip/emnist-digits",
			Self::Letters => "gzip/emnist-letters",
			Self::Mnist => "gzip/emnist-mnist",
		}
	}

	/// Number of classes in the variant.
	pub const fn classes(self) -> usize {
		match self {
			Self::Balanced | Self::ByMerge => 47,
			Self::ByClass => 62,
			Self::Digits | Self::Mnist => 10,
			Self::Letters => 26,
		}
	}
}

/// Device-resident EMNIST variant split implementing [`oa::data::Dataset`].
///
/// `images()` is `[N, 1, 28, 28]` F32 NCHW and `labels()` is `[N]` F32 class
/// indices; per-variant ranges are on [`EmnistVariant`].
pub struct Emnist {
	dataset: MatrixDataset,
	split: EmnistSplit,
	variant: EmnistVariant,
}

impl Emnist {
	/// Fetch, verify, and decode the selected variant split on the host.
	///
	/// The pack is materialized under `config.cache_dir` or the resolved
	/// dataset cache root; members already present at the manifest size are
	/// reused without refetching.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, archive decode errors when a
	/// member is missing or malformed, and an error when image and label
	/// counts disagree.
	pub fn decode(config: &EmnistConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// the callback aborts a fetch.
	pub fn decode_with_progress(
		config: &EmnistConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&EMNIST_PACK, &root, progress)?;
		let split = match config.split {
			IdxSplit::Train => "train",
			IdxSplit::Test => "test",
			IdxSplit::Validation => {
				return Err(Error::invalid_argument(
					"emnist carries no validation split",
				));
			}
		};
		let prefix = config.variant.member();
		let wanted = [
			format!("{prefix}-{split}-images-idx3-ubyte.gz"),
			format!("{prefix}-{split}-labels-idx1-ubyte.gz"),
		];
		let zip = std::fs::read(dir.join("gzip.zip"))
			.map_err(|error| Error::io("read emnist archive", error))?;
		let mut members = HashMap::new();
		extract::walk_zip(&zip, |name, bytes| {
			if wanted.iter().any(|w| w == name) {
				members.insert(name.to_owned(), bytes.to_vec());
			}
			Ok(())
		})?;
		let images_gz = members
			.remove(&wanted[0])
			.ok_or_else(|| Error::data_loss(format!("emnist archive holds no member {}", wanted[0])))?;
		let labels_gz = members
			.remove(&wanted[1])
			.ok_or_else(|| Error::data_loss(format!("emnist archive holds no member {}", wanted[1])))?;
		let (shape, mut values) = idx::decode_images(
			"emnist",
			&extract::gunzip_bytes(&images_gz)?,
			config.normalize,
		)?;
		// EMNIST IDX stores each image transposed relative to normal reading
		// orientation; transpose H/W so samples read like MNIST digits.
		let (rows, cols) = (shape[2], shape[3]);
		for image in values.chunks_exact_mut(rows * cols) {
			for r in 0..rows {
				for c in r + 1..cols {
					image.swap(r * cols + c, c * cols + r);
				}
			}
		}
		let images = MatrixData {
			name: "images",
			shape,
			values,
		};
		let (shape, values) = idx::decode_labels("emnist", &extract::gunzip_bytes(&labels_gz)?)?;
		let labels = MatrixData {
			name: "labels",
			shape,
			values,
		};
		let (count, label_count) = (images.shape[0], labels.shape[0]);
		if count != label_count {
			return Err(Error::data_loss(format!(
				"emnist split has {count} images but {label_count} labels"
			)));
		}
		Ok(Decoded {
			matrices: vec![images, labels],
			text: None,
			seq_len: 0,
		})
	}

	/// [`load_on`](Self::load_on) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`load_on`](Self::load_on), plus `Cancelled` when the
	/// callback aborts a fetch.
	pub fn load_with_progress_on(
		engine: &Engine,
		config: &EmnistConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Ok(Self {
			dataset: MatrixDataset::new(
				decoded.upload("images", engine)?,
				decoded.upload("labels", engine)?,
			)?,
			split: config.split,
			variant: config.variant,
		})
	}

	/// Full `[N, 1, 28, 28]` F32 image matrix on the owning engine.
	pub const fn images(&self) -> &Matrix {
		self.dataset.x()
	}

	/// Full `[N]` F32 label matrix on the owning engine.
	pub const fn labels(&self) -> &Matrix {
		self.dataset.y()
	}

	/// Selected split.
	pub const fn split(&self) -> EmnistSplit {
		self.split
	}

	/// Selected class grouping.
	pub const fn variant(&self) -> EmnistVariant {
		self.variant
	}
}

impl Dataset for Emnist {
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
