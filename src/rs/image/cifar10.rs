//! CIFAR-10 object recognition as a device-resident [`oa::data::Dataset`].
//!
//! The binary tarball is materialized through [`ensure_pack`](crate::pack::ensure_pack) under
//! `cifar10/1.0.0`, and each `*.bin` member is a run of 3,073-byte records:
//! one label byte followed by 1,024 red, green, and blue planar pixel bytes.
//! Upload produces `[N, 3, 32, 32]` F32 NCHW images and `[N]` F32 class
//! indices `0..=9` — the same per-sample contract as [`crate::Mnist`] but
//! color, so [`oa::data::collate`] yields `[B, 3, 32, 32]` batches.

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::decode::{Decoded, MatrixData};
use crate::extract;
use crate::idx::IdxSplit;
use crate::pack::{self, FetchProgress};

const RECORD_LEN: usize = 3_073;

#[path = "cifar10.gen.rs"]
mod generated;

pub(crate) use generated::CIFAR10_PACK;
pub use generated::{Cifar10Config, Cifar10Split};

const TRAIN_MEMBERS: &[&str] = &[
	"data_batch_1.bin",
	"data_batch_2.bin",
	"data_batch_3.bin",
	"data_batch_4.bin",
	"data_batch_5.bin",
];
const TEST_MEMBERS: &[&str] = &["test_batch.bin"];

/// Device-resident CIFAR-10 split implementing [`oa::data::Dataset`].
///
/// `images()` is `[N, 3, 32, 32]` F32 NCHW and `labels()` is `[N]` F32 class
/// indices `0..=9`.
pub struct Cifar10 {
	dataset: MatrixDataset,
	split: Cifar10Split,
}

impl Cifar10 {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and archive decode errors when
	/// a member is missing or a record run is not a multiple of 3,073 bytes.
	pub fn decode(config: &Cifar10Config) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// the callback aborts a fetch.
	pub fn decode_with_progress(
		config: &Cifar10Config,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&CIFAR10_PACK, &root, progress)?;
		let tar = extract::gunzip(&dir.join("cifar-10-binary.tar.gz"))?;
		let members = match config.split {
			IdxSplit::Train => TRAIN_MEMBERS,
			IdxSplit::Test => TEST_MEMBERS,
		};
		let mut labels = Vec::new();
		let mut pixels = Vec::new();
		for member in members {
			read_records(extract::tar_member(&tar, member)?, &mut labels, &mut pixels)?;
		}
		let scale = if config.normalize { 1.0 / 255.0 } else { 1.0 };
		let count = labels.len();
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
	/// Returns the errors of [`load_on`](Self::load_on), plus `Cancelled` when
	/// the callback aborts a fetch.
	pub fn load_with_progress_on(
		engine: &Engine,
		config: &Cifar10Config,
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
	pub const fn split(&self) -> Cifar10Split {
		self.split
	}
}

impl Dataset for Cifar10 {
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

fn read_records(member: &[u8], labels: &mut Vec<u8>, pixels: &mut Vec<u8>) -> Result<()> {
	if !member.len().is_multiple_of(RECORD_LEN) {
		return Err(Error::data_loss(format!(
			"cifar10 member holds {} bytes, not a multiple of {RECORD_LEN}",
			member.len()
		)));
	}
	let (records, _) = member.as_chunks::<RECORD_LEN>();
	for record in records {
		labels.push(record[0]);
		pixels.extend_from_slice(&record[1..]);
	}
	Ok(())
}
