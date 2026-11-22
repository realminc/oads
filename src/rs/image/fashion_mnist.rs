//! Fashion-MNIST clothing articles as a device-resident [`oa::data::Dataset`].
//!
//! The Zalando IDX payloads are materialized through [`ensure_pack`](crate::pack::ensure_pack) under
//! `fashion_mnist/1.0.0`, gunzipped and decoded on the host, and uploaded
//! once as a `[N, 1, 28, 28]` F32 image matrix and a `[N]` F32 label matrix
//! — the same sample contract as [`crate::Mnist`], so [`oa::data::collate`]
//! produces `[B, 1, 28, 28]` NCHW batches.

use std::path::PathBuf;

use oa::data::{Dataset, Sample};
use oa::matrix::{self, Matrix};
use oa::{Engine, Error, Result};

use crate::decode::{Decoded, MatrixData};
use crate::extract;
use crate::idx::{self, IdxSplit};
use crate::pack::{self, FetchProgress, Pack, PackFile};

pub(crate) const FASHION_MNIST_PACK: Pack = Pack {
	name: "fashion_mnist",
	version: "1.0.0",
	license: "MIT",
	files: &[
		PackFile {
			name: "train-images-idx3-ubyte.gz",
			size: 26_421_880,
			shake256: "287217197fbade1cabc218dbc30a6ba2da0e6a3af5b4aa4185f3fd24f94ddabf",
			urls: &[
				"http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
				"https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz",
			],
		},
		PackFile {
			name: "train-labels-idx1-ubyte.gz",
			size: 29_515,
			shake256: "8acdf6a86fbee77624e8f5746b475a5bf3f387f76370df41cd7d6b63cea2fa21",
			urls: &[
				"http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
				"https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz",
			],
		},
		PackFile {
			name: "t10k-images-idx3-ubyte.gz",
			size: 4_422_102,
			shake256: "897f33e93438e9eb054bd6103c0396f009a7d1035be8c726ef28ca3bd8dfabed",
			urls: &[
				"http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
				"https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz",
			],
		},
		PackFile {
			name: "t10k-labels-idx1-ubyte.gz",
			size: 5_148,
			shake256: "16a6ca6f5f7f62ce4ba3d742a4fa9444a73b82bd5e5df971edd4ea468f6d77cf",
			urls: &[
				"http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
				"https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz",
			],
		},
	],
};

/// Official Fashion-MNIST split selection.
pub type FashionMnistSplit = IdxSplit;

/// Load options for [`FashionMnist::load`].
#[derive(Clone, Debug)]
pub struct FashionMnistConfig {
	/// Which official split to materialize.
	pub split: FashionMnistSplit,
	/// Cache root override. `None` resolves `OA_DATA_DIR`, then the platform
	/// cache directory; see [`data_root`](crate::pack::data_root).
	pub cache_dir: Option<PathBuf>,
	/// Scale pixels into `[0, 1]`. `false` keeps raw `0..=255` values.
	pub normalize: bool,
}

impl Default for FashionMnistConfig {
	fn default() -> Self {
		Self {
			split: IdxSplit::Train,
			cache_dir: None,
			normalize: true,
		}
	}
}

/// Device-resident Fashion-MNIST split implementing [`oa::data::Dataset`].
///
/// `images()` is `[N, 1, 28, 28]` F32 NCHW and `labels()` is `[N]` F32 class
/// indices `0..=9`.
pub struct FashionMnist {
	images: Matrix,
	labels: Matrix,
	split: FashionMnistSplit,
}

impl FashionMnist {
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
	pub fn decode(config: &FashionMnistConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// the callback aborts a fetch.
	pub fn decode_with_progress(
		config: &FashionMnistConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&FASHION_MNIST_PACK, &root, progress)?;
		let (images_name, labels_name) = match config.split {
			IdxSplit::Train => ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
			IdxSplit::Test => ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
		};
		let (shape, values) = idx::decode_images(
			"fashion_mnist",
			&extract::gunzip(&dir.join(images_name))?,
			config.normalize,
		)?;
		let images = MatrixData {
			name: "images",
			shape,
			values,
		};
		let (shape, values) =
			idx::decode_labels("fashion_mnist", &extract::gunzip(&dir.join(labels_name))?)?;
		let labels = MatrixData {
			name: "labels",
			shape,
			values,
		};
		let (count, label_count) = (images.shape[0], labels.shape[0]);
		if count != label_count {
			return Err(Error::data_loss(format!(
				"fashion_mnist split has {count} images but {label_count} labels"
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
	pub fn load(engine: &Engine, config: &FashionMnistConfig) -> Result<Self> {
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
		config: &FashionMnistConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Ok(Self {
			images: decoded.upload("images", engine)?,
			labels: decoded.upload("labels", engine)?,
			split: config.split,
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
	pub const fn split(&self) -> FashionMnistSplit {
		self.split
	}

	fn check_index(&self, index: i64) -> Result<()> {
		if index < 0 || index >= self.len() {
			return Err(Error::out_of_range(
				"fashion_mnist index is outside the dataset",
			));
		}
		Ok(())
	}
}

impl Dataset for FashionMnist {
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
