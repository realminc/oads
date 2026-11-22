//! Yelp Review Polarity sentiment corpus as a device-resident
//! [`oa::data::Dataset`].
//!
//! The CSV tarball is materialized through
//! [`ensure_pack`](crate::pack::ensure_pack) under `yelp_polarity/1.0.0`.
//! Each review encodes as a `[text_len]` F32 UTF-8 byte window, truncated
//! or zero-padded; `labels()` is `[N]` F32 class indices `0..=1`
//! (negative/positive — raw CSV labels `1..=2` minus one).

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Result};

use crate::csv;
use crate::decode::Decoded;
use crate::extract;
use crate::idx::IdxSplit;
use crate::pack::{self, FetchProgress};
use crate::text::document;

#[path = "yelp_polarity.gen.rs"]
mod generated;

pub(crate) use generated::YELP_POLARITY_PACK;
pub use generated::{YelpPolarityConfig, YelpPolaritySplit};

/// Device-resident Yelp Polarity split implementing [`oa::data::Dataset`].
///
/// `text()` is `[N, text_len]` F32 byte windows and `labels()` is `[N]` F32
/// class indices `0..=1`.
pub struct YelpPolarity {
	dataset: MatrixDataset,
	split: YelpPolaritySplit,
	text_len: usize,
}

impl YelpPolarity {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, archive decode errors, and
	/// CSV parse errors for malformed rows.
	pub fn decode(config: &YelpPolarityConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// the callback aborts a fetch.
	pub fn decode_with_progress(
		config: &YelpPolarityConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&YELP_POLARITY_PACK, &root, progress)?;
		let tar = extract::gunzip(&dir.join("yelp_review_polarity_csv.tgz"))?;
		let member = match config.split {
			IdxSplit::Train => "train.csv",
			IdxSplit::Test => "test.csv",
		};
		let records = csv::parse("yelp_polarity", extract::tar_member(&tar, member)?)?;
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
		config: &YelpPolarityConfig,
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
	pub const fn split(&self) -> YelpPolaritySplit {
		self.split
	}

	/// Document window length.
	pub const fn text_len(&self) -> usize {
		self.text_len
	}
}

impl Dataset for YelpPolarity {
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
