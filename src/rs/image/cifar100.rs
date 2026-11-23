//! CIFAR-100 labeled color images as a device-resident [`oa::data::Dataset`].
//!
//! The binary archive is materialized through [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `cifar100/1.0.0`, decoded on the host, and uploaded once as a
//! `[N, 3, 32, 32]` F32 image matrix and a `[N]` F32 label matrix holding the
//! fine label index `0..=99`. `get_item` returns the `[3, 32, 32]` CHW slice
//! so [`oa::data::collate`] produces `[B, 3, 32, 32]` NCHW batches.

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::extract;
use crate::core::idx::IdxSplit;
use crate::core::pack::{self, FetchProgress};

const RECORD_LEN: usize = 3_074;

#[path = "cifar100.gen.rs"]
mod generated;

pub(crate) use generated::CIFAR100_PACK;
pub use generated::{Cifar100Config, Cifar100Split};

/// Device-resident CIFAR-100 split implementing [`oa::data::Dataset`].
///
/// `images()` is `[N, 3, 32, 32]` F32 NCHW and `labels()` is `[N]` F32 fine
/// class indices `0..=99`. The coarse `0..=19` superclass byte in each record
/// is not part of the dataset contract.
pub struct Cifar100 {
	dataset: MatrixDataset,
	split: Cifar100Split,
}

impl Cifar100 {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// The pack is materialized under `config.cache_dir` or the resolved
	/// dataset cache root; members already present at the manifest size are
	/// reused without refetching.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, archive decode errors when a
	/// member is truncated or its record count is not integral, and an error
	/// when image and label counts disagree.
	pub fn decode(config: &Cifar100Config) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// the callback aborts a fetch.
	pub fn decode_with_progress(
		config: &Cifar100Config,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&CIFAR100_PACK, &root, progress)?;
		let tar = extract::gunzip(&dir.join("cifar-100-binary.tar.gz"))?;
		let member = match config.split {
			IdxSplit::Train => "train.bin",
			IdxSplit::Test => "test.bin",
			IdxSplit::Validation => {
				return Err(Error::invalid_argument(
					"cifar100 carries no validation split",
				));
			}
		};
		let records = extract::tar_member(&tar, member)?;
		if records.len() % RECORD_LEN != 0 {
			return Err(Error::data_loss(format!(
				"cifar100 {member} is {} bytes, not a multiple of {RECORD_LEN}",
				records.len()
			)));
		}
		let count = records.len() / RECORD_LEN;
		let mut labels = Vec::with_capacity(count);
		let mut pixels = Vec::with_capacity(count * 3_072);
		for record in records.as_chunks::<RECORD_LEN>().0 {
			labels.push(record[1]);
			pixels.extend_from_slice(&record[2..]);
		}
		let scale = if config.normalize { 1.0 / 255.0 } else { 1.0 };
		Ok(Decoded {
			matrices: vec![
				MatrixData {
					name: "images",
					shape: vec![count, 3, 32, 32],
					values: pixels
						.iter()
						.map(|&v| f32::from(v) * scale)
						.collect::<Vec<_>>(),
				},
				MatrixData {
					name: "labels",
					shape: vec![count],
					values: labels.iter().map(|&v| f32::from(v)).collect::<Vec<_>>(),
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
	/// Returns the errors of [`load_on`](Self::load_on), plus `Cancelled` when the
	/// callback aborts a fetch.
	pub fn load_with_progress_on(
		engine: &Engine,
		config: &Cifar100Config,
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

	/// Full `[N, 3, 32, 32]` F32 image matrix on the owning engine.
	pub const fn images(&self) -> &Matrix {
		self.dataset.x()
	}

	/// Full `[N]` F32 fine label matrix on the owning engine.
	pub const fn labels(&self) -> &Matrix {
		self.dataset.y()
	}

	/// Selected split.
	pub const fn split(&self) -> Cifar100Split {
		self.split
	}
}

impl Dataset for Cifar100 {
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
