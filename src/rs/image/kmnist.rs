//! Kuzushiji-MNIST cursive Japanese characters as a device-resident [`oa::data::Dataset`].
//!
//! The official IDX payloads are materialized through [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `kmnist/1.0.0`, gunzipped and decoded on the host, and uploaded once as a
//! `[N, 1, 28, 28]` F32 image matrix and a `[N]` F32 label matrix.
//! [`oa::data::Dataset::get_item`] returns the `[1, 28, 28]` CHW slice so
//! [`oa::data::collate`] produces `[B, 1, 28, 28]` NCHW batches.

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::extract;
use crate::core::idx::{self, IdxSplit};
use crate::core::pack::{self, FetchProgress};

#[path = "kmnist.gen.rs"]
mod generated;

pub(crate) use generated::KMNIST_PACK;
pub use generated::{KMnistConfig, KMnistSplit};

/// Device-resident KMNIST split implementing [`oa::data::Dataset`].
///
/// `images()` is `[N, 1, 28, 28]` F32 NCHW and `labels()` is `[N]` F32 class
/// indices `0..=9` (o, ki, su, tsu, na, ha, ma, ya, re, wo).
pub struct KMnist {
	dataset: MatrixDataset,
	split: KMnistSplit,
}

impl KMnist {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// The pack is materialized under `config.cache_dir` or the resolved
	/// dataset cache root; members already present at the manifest size are
	/// reused without refetching.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, IDX decode errors when a member
	/// is truncated or malformed, and an error when image and label counts
	/// disagree.
	pub fn decode(config: &KMnistConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// the callback aborts a fetch.
	pub fn decode_with_progress(
		config: &KMnistConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&KMNIST_PACK, &root, progress)?;
		let (images_name, labels_name) = match config.split {
			IdxSplit::Train => ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
			IdxSplit::Test => ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
			IdxSplit::Validation => {
				return Err(Error::invalid_argument(
					"kmnist carries no validation split",
				));
			}
		};
		let (shape, values) = idx::decode_images(
			"kmnist",
			&extract::gunzip(&dir.join(images_name))?,
			config.normalize,
		)?;
		let images = MatrixData {
			name: "images",
			shape,
			values,
		};
		let (shape, values) = idx::decode_labels("kmnist", &extract::gunzip(&dir.join(labels_name))?)?;
		let labels = MatrixData {
			name: "labels",
			shape,
			values,
		};
		let (count, label_count) = (images.shape[0], labels.shape[0]);
		if count != label_count {
			return Err(Error::data_loss(format!(
				"kmnist split has {count} images but {label_count} labels"
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
	/// Returns the errors of [`load_on`](Self::load_on), plus `Cancelled` when
	/// the callback aborts a fetch.
	pub fn load_with_progress_on(
		engine: &Engine,
		config: &KMnistConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Self::new(
			decoded.upload("images", engine)?,
			decoded.upload("labels", engine)?,
			config.split,
		)
	}

	/// Wrap caller-owned matrices as a KMnist dataset.
	///
	/// `images` must be rank-4 `[N, C, H, W]` F32 and `labels` rank-1 `[N]`
	/// F32 on the same engine.
	///
	/// # Errors
	///
	/// Returns an error for a non-rank-4 image matrix, plus the count,
	/// dtype, and engine errors of [`MatrixDataset::new`].
	pub fn new(images: Matrix, labels: Matrix, split: KMnistSplit) -> Result<Self> {
		if images.shape().len() != 4 {
			return Err(Error::invalid_argument(
				"kmnist requires [N, C, H, W] images",
			));
		}
		Ok(Self {
			dataset: MatrixDataset::new(images, labels)?,
			split,
		})
	}

	/// Full `[N, 1, 28, 28]` F32 image matrix on the owning engine.
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
	pub const fn split(&self) -> KMnistSplit {
		self.split
	}
}

impl Dataset for KMnist {
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
