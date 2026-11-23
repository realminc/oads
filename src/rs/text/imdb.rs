//! IMDb large movie review sentiment corpus as a device-resident
//! [`oa::data::Dataset`].
//!
//! The `aclImdb` tarball is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `imdb/1.0.0`. Review text
//! under `<split>/{pos,neg}/*.txt` encodes as a `[text_len]` F32 UTF-8 byte
//! window, truncated or zero-padded; `labels()` is `[N]` F32 class indices
//! `0..=1` (neg/pos). `unsup` members are ignored.

use std::collections::BTreeMap;

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::core::decode::Decoded;
use crate::core::extract;
use crate::core::idx::IdxSplit;
use crate::core::pack::{self, FetchProgress};
use crate::text::document;

#[path = "imdb.gen.rs"]
mod generated;

pub(crate) use generated::IMDB_PACK;
pub use generated::{ImdbConfig, ImdbSplit};

/// Device-resident IMDb split implementing [`oa::data::Dataset`].
///
/// `text()` is `[N, text_len]` F32 byte windows and `labels()` is `[N]` F32
/// class indices `0..=1`.
pub struct Imdb {
	dataset: MatrixDataset,
	split: ImdbSplit,
	text_len: usize,
}

impl Imdb {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and archive decode errors.
	pub fn decode(config: &ImdbConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// the callback aborts a fetch.
	pub fn decode_with_progress(
		config: &ImdbConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&IMDB_PACK, &root, progress)?;
		let tar = extract::gunzip(&dir.join("aclImdb_v1.tar.gz"))?;
		let split = match config.split {
			IdxSplit::Train => "train",
			IdxSplit::Test => "test",
			IdxSplit::Validation => {
				return Err(Error::invalid_argument("imdb carries no validation split"));
			}
		};
		// Sorted member order keeps document order deterministic: pos before
		// neg would interleave differently across tar listings; BTreeMap gives
		// a stable lexicographic pass per class directory.
		let mut documents: BTreeMap<String, (f32, Vec<u8>)> = BTreeMap::new();
		extract::walk_tar(&tar, |name, bytes| {
			if let Some(class) = name
				.strip_prefix(&format!("aclImdb/{split}/pos/"))
				.map(|_| 1.0_f32)
				.or_else(|| {
					name
						.strip_prefix(&format!("aclImdb/{split}/neg/"))
						.map(|_| 0.0_f32)
				}) {
				documents.insert(name.to_owned(), (class, bytes.to_vec()));
			}
			Ok(())
		})?;
		let documents: Vec<(f32, Vec<u8>)> = documents.into_values().collect();
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
		config: &ImdbConfig,
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
	pub const fn split(&self) -> ImdbSplit {
		self.split
	}

	/// Document window length.
	pub const fn text_len(&self) -> usize {
		self.text_len
	}
}

impl Dataset for Imdb {
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
