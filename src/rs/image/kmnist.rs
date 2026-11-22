//! Kuzushiji-MNIST Japanese cursive characters as a device-resident
//! [`oa::data::Dataset`].
//!
//! The ROIS-CDS IDX payloads are materialized through [`ensure_pack`](crate::pack::ensure_pack) under
//! `kmnist/1.0.0`, gunzipped and decoded on the host, and uploaded once as a
//! `[N, 1, 28, 28]` F32 image matrix and a `[N]` F32 label matrix — the same
//! sample contract as [`crate::Mnist`], so [`oa::data::collate`] produces
//! `[B, 1, 28, 28]` NCHW batches.

use std::path::PathBuf;

use oa::data::{Dataset, Sample};
use oa::matrix::{self, Matrix};
use oa::{Engine, Error, Result};

use crate::decode::{Decoded, MatrixData};
use crate::extract;
use crate::idx::{self, IdxSplit};
use crate::pack::{self, FetchProgress, Pack, PackFile};

pub(crate) const KMNIST_PACK: Pack = Pack {
	name: "kmnist",
	version: "1.0.0",
	license: "CC-BY-SA-4.0",
	files: &[
		PackFile {
			name: "train-images-idx3-ubyte.gz",
			size: 18_165_135,
			shake256: "514a63d19287b9363d1259a7c5a178a75b82322fbe90a9ac9bf240734250006b",
			urls: &["http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz"],
		},
		PackFile {
			name: "train-labels-idx1-ubyte.gz",
			size: 29_497,
			shake256: "271b9566abb706d1197a89a9940d3ea709ba49a9dc6a3dae7543f1ebf29b58b5",
			urls: &["http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz"],
		},
		PackFile {
			name: "t10k-images-idx3-ubyte.gz",
			size: 3_041_136,
			shake256: "b8c5f3e82249027924131038bea598300a7a2639c411d3f2f79500a7667cf7bf",
			urls: &["http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz"],
		},
		PackFile {
			name: "t10k-labels-idx1-ubyte.gz",
			size: 5_120,
			shake256: "e834280bf6aefdd90293b34ead5e67b1e0c30cac93f1ed49ed500f6c7e3fa1d5",
			urls: &["http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz"],
		},
	],
};

/// Official KMNIST split selection.
pub type KMnistSplit = IdxSplit;

/// Load options for [`KMnist::load`].
#[derive(Clone, Debug)]
pub struct KMnistConfig {
	/// Which official split to materialize.
	pub split: KMnistSplit,
	/// Cache root override. `None` resolves `OA_DATA_DIR`, then the platform
	/// cache directory; see [`data_root`](crate::pack::data_root).
	pub cache_dir: Option<PathBuf>,
	/// Scale pixels into `[0, 1]`. `false` keeps raw `0..=255` values.
	pub normalize: bool,
}

impl Default for KMnistConfig {
	fn default() -> Self {
		Self {
			split: IdxSplit::Train,
			cache_dir: None,
			normalize: true,
		}
	}
}

/// Device-resident KMNIST split implementing [`oa::data::Dataset`].
///
/// `images()` is `[N, 1, 28, 28]` F32 NCHW and `labels()` is `[N]` F32 class
/// indices `0..=9` (10 Kuzushiji characters).
pub struct KMnist {
	images: Matrix,
	labels: Matrix,
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

	/// Decode the selected split and upload it to `engine`.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, IDX decode errors, and matrix
	/// upload errors.
	pub fn load(engine: &Engine, config: &KMnistConfig) -> Result<Self> {
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
		config: &KMnistConfig,
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
	pub const fn split(&self) -> KMnistSplit {
		self.split
	}

	fn check_index(&self, index: i64) -> Result<()> {
		if index < 0 || index >= self.len() {
			return Err(Error::out_of_range("kmnist index is outside the dataset"));
		}
		Ok(())
	}
}

impl Dataset for KMnist {
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
