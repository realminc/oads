//! Yelp Review Full five-star sentiment corpus as a device-resident
//! [`oa::data::Dataset`].
//!
//! The CSV tarball is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `yelp_review_full/1.0.0`.
//! Each review encodes as a `[text_len]` F32 UTF-8 byte window, truncated
//! or zero-padded; `labels()` is `[N]` F32 class indices `0..=4`
//! (one through five stars — raw CSV labels `1..=5` minus one).

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::core::csv;
use crate::core::decode::Decoded;
use crate::core::extract;
use crate::core::idx::IdxSplit;
use crate::core::pack::{self, FetchProgress};
use crate::text::document;

#[path = "yelp_review_full.gen.rs"]
mod generated;

pub(crate) use generated::YELP_REVIEW_FULL_PACK;
pub use generated::{YelpReviewFullConfig, YelpReviewFullSplit};

/// Device-resident Yelp Polarity split implementing [`oa::data::Dataset`].
///
/// `text()` is `[N, text_len]` F32 byte windows and `labels()` is `[N]` F32
/// class indices `0..=4`.
pub struct YelpReviewFull {
	dataset: MatrixDataset,
	split: YelpReviewFullSplit,
	text_len: usize,
}

impl YelpReviewFull {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, archive decode errors, and
	/// CSV parse errors for malformed rows.
	pub fn decode(config: &YelpReviewFullConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// the callback aborts a fetch.
	pub fn decode_with_progress(
		config: &YelpReviewFullConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&YELP_REVIEW_FULL_PACK, &root, progress)?;
		let tar = extract::gunzip(&dir.join("yelp_review_full_csv.tgz"))?;
		let member = match config.split {
			IdxSplit::Train => "train.csv",
			IdxSplit::Test => "test.csv",
			IdxSplit::Validation => {
				return Err(Error::invalid_argument(
					"yelp_review_full carries no validation split",
				));
			}
		};
		let records = csv::parse("yelp_review_full", extract::tar_member(&tar, member)?)?;
		let mut documents = Vec::with_capacity(records.len());
		for record in records {
			if record.len() < 2 {
				continue;
			}
			let label = record[0].parse::<f32>().unwrap_or(0.0) - 1.0;
			documents.push((label, record[1].clone().into_bytes()));
		}
		let (text, labels) = document::encode("text", &documents, config.text_len, config.normalize)?;
		Ok(Decoded {
			matrices: vec![text, labels],
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
		config: &YelpReviewFullConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Ok(Self {
			dataset: MatrixDataset::new(
				decoded.upload("text", engine)?,
				decoded.upload("labels", engine)?,
			)?,
			split: config.split,
			text_len: config.text_len,
		})
	}

	/// Full `[N, text_len]` F32 byte-window matrix on the owning engine.
	pub const fn text(&self) -> &Matrix {
		self.dataset.x()
	}

	/// Full `[N]` F32 label matrix on the owning engine.
	pub const fn labels(&self) -> &Matrix {
		self.dataset.y()
	}

	/// Selected split.
	pub const fn split(&self) -> YelpReviewFullSplit {
		self.split
	}

	/// Document window length.
	pub const fn text_len(&self) -> usize {
		self.text_len
	}
}

impl Dataset for YelpReviewFull {
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
