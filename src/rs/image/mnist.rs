//! MNIST handwritten digits as a device-resident [`oa::data::Dataset`].
//!
//! The official IDX payloads are materialized through [`ensure_pack`](crate::pack::ensure_pack) under
//! `mnist/1.0.0`, gunzipped and decoded on the host, and uploaded once as a
//! `[N, 1, 28, 28]` F32 image matrix and a `[N]` F32 label matrix.
//! [`oa::data::Dataset::get_item`] returns the `[1, 28, 28]` CHW slice so
//! [`oa::data::collate`] produces `[B, 1, 28, 28]` NCHW batches.

use std::path::PathBuf;

use oa::data::{Dataset, Sample};
use oa::matrix::{self, Matrix};
use oa::{Engine, Error, Result};

use crate::decode::{Decoded, MatrixData};
use crate::extract;
use crate::idx::{self, IdxSplit};
use crate::pack::{self, FetchProgress, Pack, PackFile};

pub(crate) const MNIST_PACK: Pack = Pack {
	name: "mnist",
	version: "1.0.0",
	license: "CC-BY-SA-3.0",
	files: &[
		PackFile {
			name: "train-images-idx3-ubyte.gz",
			size: 9_912_422,
			shake256: "1314ff889cdcbd8edcfad0916f94202767b7258e93ab75534affc2ccbb6fa575",
			urls: &[
				"https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
				"https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
			],
		},
		PackFile {
			name: "train-labels-idx1-ubyte.gz",
			size: 28_881,
			shake256: "6bcc237fd21a81c96ff200bdae886930ad9b9576fcd16bc49038f5ae7e275c01",
			urls: &[
				"https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
				"https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
			],
		},
		PackFile {
			name: "t10k-images-idx3-ubyte.gz",
			size: 1_648_877,
			shake256: "fb5dd7142cba2dad6230d7b75b31917b34294280b67b60d7b95bc2e1e1dcd671",
			urls: &[
				"https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
				"https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
			],
		},
		PackFile {
			name: "t10k-labels-idx1-ubyte.gz",
			size: 4_542,
			shake256: "105e631874f55e52bafd7c85ab4be9152f8296fc7aaa475815d8f56f654faea7",
			urls: &[
				"https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
				"https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
			],
		},
	],
};

/// Official MNIST split selection.
pub type MnistSplit = IdxSplit;

/// Load options for [`Mnist::load`].
#[derive(Clone, Debug)]
pub struct MnistConfig {
	/// Which official split to materialize.
	pub split: MnistSplit,
	/// Cache root override. `None` resolves `OA_DATA_DIR`, then the platform
	/// cache directory; see [`data_root`](crate::pack::data_root).
	pub cache_dir: Option<PathBuf>,
	/// Scale pixels into `[0, 1]`. `false` keeps raw `0..=255` values.
	pub normalize: bool,
}

impl Default for MnistConfig {
	fn default() -> Self {
		Self {
			split: MnistSplit::Train,
			cache_dir: None,
			normalize: true,
		}
	}
}

/// Device-resident MNIST split implementing [`oa::data::Dataset`].
///
/// `images()` is `[N, 1, 28, 28]` F32 NCHW and `labels()` is `[N]` F32 class
/// indices `0..=9`.
pub struct Mnist {
	images: Matrix,
	labels: Matrix,
	split: MnistSplit,
}

impl Mnist {
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
	pub fn decode(config: &MnistConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// the callback aborts a fetch.
	pub fn decode_with_progress(
		config: &MnistConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&MNIST_PACK, &root, progress)?;
		let (images_name, labels_name) = match config.split {
			IdxSplit::Train => ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
			IdxSplit::Test => ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
		};
		let (shape, values) = idx::decode_images(
			"mnist",
			&extract::gunzip(&dir.join(images_name))?,
			config.normalize,
		)?;
		let images = MatrixData {
			name: "images",
			shape,
			values,
		};
		let (shape, values) = idx::decode_labels("mnist", &extract::gunzip(&dir.join(labels_name))?)?;
		let labels = MatrixData {
			name: "labels",
			shape,
			values,
		};
		let (count, label_count) = (images.shape[0], labels.shape[0]);
		if count != label_count {
			return Err(Error::data_loss(format!(
				"mnist split has {count} images but {label_count} labels"
			)));
		}
		Ok(Decoded {
			matrices: vec![images, labels],
			text: None,
			seq_len: 0,
		})
	}

	/// Decode the selected split and upload it to `engine`.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, IDX decode errors, and matrix
	/// upload errors.
	pub fn load(engine: &Engine, config: &MnistConfig) -> Result<Self> {
		Self::load_with_progress(engine, config, &mut |_| true)
	}

	/// [`load`](Self::load) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`load`](Self::load), plus `Cancelled` when
	/// the callback aborts a fetch.
	pub fn load_with_progress(
		engine: &Engine,
		config: &MnistConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Ok(Self {
			images: decoded.upload("images", engine)?,
			labels: decoded.upload("labels", engine)?,
			split: config.split,
		})
	}

	/// Wrap caller-owned matrices as an MNIST dataset.
	///
	/// `images` must be rank-4 `[N, C, H, W]` F32 and `labels` rank-1 `[N]`
	/// F32 on the same engine.
	///
	/// # Errors
	///
	/// Returns an error for mismatched counts, wrong ranks, wrong dtypes, or
	/// matrices owned by different engines.
	pub fn new(images: Matrix, labels: Matrix, split: MnistSplit) -> Result<Self> {
		if images.shape().len() != 4
			|| labels.shape().len() != 1
			|| images.shape()[0] != labels.shape()[0]
		{
			return Err(Error::invalid_argument(
				"mnist requires [N, C, H, W] images and [N] labels with equal counts",
			));
		}
		if images.dtype() != oa::DType::F32 || labels.dtype() != oa::DType::F32 {
			return Err(Error::invalid_argument(
				"mnist requires F32 images and labels",
			));
		}
		if !images.same_engine(&labels) {
			return Err(Error::invalid_argument(
				"mnist images and labels must share one engine",
			));
		}
		Ok(Self {
			images,
			labels,
			split,
		})
	}

	/// Full `[N, 1, 28, 28]` F32 image matrix on the owning engine.
	#[must_use]
	pub const fn images(&self) -> &Matrix {
		&self.images
	}

	/// Full `[N]` F32 label matrix on the owning engine.
	#[must_use]
	pub const fn labels(&self) -> &Matrix {
		&self.labels
	}

	/// The split this dataset was materialized from.
	#[must_use]
	pub const fn split(&self) -> MnistSplit {
		self.split
	}

	fn check_index(&self, index: i64) -> Result<()> {
		if index < 0 || index >= self.len() {
			return Err(Error::out_of_range("mnist index is outside the dataset"));
		}
		Ok(())
	}
}

impl Dataset for Mnist {
	fn len(&self) -> i64 {
		i64::try_from(self.labels.shape()[0]).unwrap_or(i64::MAX)
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		self.check_index(index)?;
		matrix::slice(&self.images, 0, index, index + 1)?.squeeze(0)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		self.check_index(index)?;
		let x = matrix::slice(&self.images, 0, index, index + 1)?.squeeze(0)?;
		let y = matrix::slice(&self.labels, 0, index, index + 1)?;
		Ok(Sample::new_with_label(x, y))
	}
}
