//! AG News topic-classification corpus as a device-resident
//! [`oa::data::Dataset`].
//!
//! The CSV tarball is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `ag_news/1.0.0`. Each row
//! encodes as a `[text_len]` F32 UTF-8 byte window over `title` + newline +
//! `description`, truncated or zero-padded; `labels()` is `[N]` F32 class
//! indices `0..=3` (World, Sports, Business, Sci/Tech — raw CSV labels
//! `1..=4` minus one). `get_item` returns the `[text_len]` row so
//! [`oa::data::collate`] produces `[B, text_len]` batches.

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::core::csv;
use crate::core::decode::Decoded;
use crate::core::extract;
use crate::core::idx::IdxSplit;
use crate::core::pack::{self, FetchProgress};
use crate::text::document;

#[path = "ag_news.gen.rs"]
mod generated;

pub(crate) use generated::AG_NEWS_PACK;
pub use generated::{AgNewsConfig, AgNewsSplit};

/// Device-resident AG News split implementing [`oa::data::Dataset`].
///
/// `text()` is `[N, text_len]` F32 byte windows and `labels()` is `[N]` F32
/// class indices `0..=3`.
pub struct AgNews {
	dataset: MatrixDataset,
	split: AgNewsSplit,
	text_len: usize,
}

impl AgNews {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, archive decode errors, and
	/// CSV parse errors for malformed rows.
	pub fn decode(config: &AgNewsConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// the callback aborts a fetch.
	pub fn decode_with_progress(
		config: &AgNewsConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&AG_NEWS_PACK, &root, progress)?;
		let tar = extract::gunzip(&dir.join("ag_news_csv.tgz"))?;
		let member = match config.split {
			IdxSplit::Train => "train.csv",
			IdxSplit::Test => "test.csv",
			IdxSplit::Validation => {
				return Err(Error::invalid_argument(
					"ag_news carries no validation split",
				));
			}
		};
		let records = csv::parse("ag_news", extract::tar_member(&tar, member)?)?;
		let mut documents = Vec::with_capacity(records.len());
		for record in records {
			if record.len() < 3 {
				continue;
			}
			let label = record[0].parse::<f32>().unwrap_or(0.0) - 1.0;
			let mut text = record[1].clone().into_bytes();
			text.push(b'\n');
			text.extend_from_slice(record[2].as_bytes());
			documents.push((label, text));
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
		config: &AgNewsConfig,
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
	pub const fn split(&self) -> AgNewsSplit {
		self.split
	}

	/// Document window length.
	pub const fn text_len(&self) -> usize {
		self.text_len
	}
}

impl Dataset for AgNews {
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
