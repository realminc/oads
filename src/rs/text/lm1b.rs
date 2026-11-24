//! One Billion Word language-modeling benchmark as a sentence stream.
//!
//! `lm-benchmark.tar.gz` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `lm1b/1.0.0`.
//! The `training-monolingual.tokenized.shuffled` shards carry one
//! tokenized sentence per line; `heldout-monolingual.tokenized.shuffled`
//! maps to Validation. Sentences flatten to a `[B]` F32 byte blob with a
//! `[N]` `sentence_lengths` matrix (offsets exceed F32 integer precision
//! past 16 MiB, so callers derive spans by cumulative sum). `get_item`
//! returns the `[1, len_i]` sentence row for the requested index.

use oa::data::{Dataset, Sample};
use oa::matrix::{self, Matrix};
use oa::{Engine, Error, Result};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::extract;
use crate::core::pack::{self, FetchProgress};

#[path = "lm1b.gen.rs"]
mod generated;

pub(crate) use generated::LM1B_PACK;
pub use generated::{Lm1bConfig, Lm1bSplit};

fn prefix(split: Lm1bSplit) -> &'static str {
	match split {
		Lm1bSplit::Train => "training-monolingual.tokenized.shuffled/news.en-",
		Lm1bSplit::Validation => "heldout-monolingual.tokenized.shuffled/news.en.heldout-",
	}
}

/// Flatten every sentence line under `prefix` into a byte blob plus a
/// per-sentence length table and host prefix-sum index.
fn decode_members(
	pack: &str,
	tar_path: &std::path::Path,
	prefix: &str,
	count: Option<usize>,
) -> Result<(Decoded, Vec<u64>)> {
	let mut blob: Vec<u8> = Vec::new();
	let mut lengths: Vec<f32> = Vec::new();
	let mut offsets: Vec<u64> = Vec::new();
	extract::walk_tar_gz_reader(tar_path, |name, member| {
		if !name.contains(prefix) {
			return Ok(());
		}
		use std::io::BufRead;
		for line in std::io::BufReader::with_capacity(1 << 20, member).lines() {
			let line = line.map_err(|error| Error::io("read lm1b sentence", error))?;
			if line.is_empty() {
				continue;
			}
			if count.is_some_and(|limit| lengths.len() >= limit) {
				return Err(Error::cancelled(format!("{pack} row cap reached")));
			}
			offsets.push(blob.len() as u64);
			blob.extend_from_slice(line.as_bytes());
			lengths.push(line.len() as f32);
		}
		Ok(())
	})?;
	if offsets.is_empty() {
		return Err(Error::data_loss(format!(
			"{pack} archive lacks {prefix} members"
		)));
	}
	let sentences = lengths.len();
	Ok((
		Decoded {
			matrices: vec![
				MatrixData {
					name: "sentence",
					shape: vec![blob.len()],
					values: blob.iter().map(|&byte| f32::from(byte)).collect(),
				},
				MatrixData {
					name: "sentence_lengths",
					shape: vec![sentences],
					values: lengths,
				},
			],
			text: None,
			seq_len: 0,
		},
		offsets,
	))
}

/// Lm1b sentence stream: a flat `[B]` byte blob plus `[N]` lengths.
pub struct Lm1b {
	sentence: Matrix,
	sentence_lengths: Matrix,
	offsets: Vec<u64>,
	rows: usize,
}

impl Lm1b {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and member decode errors.
	pub fn decode(config: &Lm1bConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus
	/// `Cancelled` when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &Lm1bConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&LM1B_PACK, &root, progress)?;
		let (decoded, _) = decode_members(
			"lm1b",
			&dir.join("lm-benchmark.tar.gz"),
			prefix(config.split),
			config.count,
		)?;
		Ok(decoded)
	}

	/// [`load_on`](Self::load_on) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`load_on`](Self::load_on), plus
	/// `Cancelled` when the callback aborts a fetch.
	pub fn load_with_progress_on(
		engine: &Engine,
		config: &Lm1bConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&LM1B_PACK, &root, progress)?;
		let (decoded, offsets) = decode_members(
			"lm1b",
			&dir.join("lm-benchmark.tar.gz"),
			prefix(config.split),
			config.count,
		)?;
		Ok(Self {
			rows: offsets.len(),
			sentence: decoded.upload("sentence", engine)?,
			sentence_lengths: decoded.upload("sentence_lengths", engine)?,
			offsets,
		})
	}

	/// `[B]` F32 flat sentence byte blob on the owning engine.
	#[must_use]
	pub const fn sentence(&self) -> &Matrix {
		&self.sentence
	}

	/// `[N]` F32 per-sentence byte lengths; cumulative sums give blob
	/// offsets.
	#[must_use]
	pub const fn sentence_lengths(&self) -> &Matrix {
		&self.sentence_lengths
	}
}

impl Dataset for Lm1b {
	fn len(&self) -> i64 {
		i64::try_from(self.rows).unwrap_or(i64::MAX)
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		let start = *self
			.offsets
			.get(index as usize)
			.ok_or_else(|| Error::invalid_argument("lm1b index out of range"))?;
		let end = self
			.offsets
			.get(index as usize + 1)
			.copied()
			.unwrap_or(self.sentence.shape()[0] as u64);
		matrix::slice(
			&self.sentence,
			0,
			i64::try_from(start).unwrap_or(i64::MAX),
			i64::try_from(end).unwrap_or(i64::MAX),
		)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		Ok(Sample::new(self.get_item(index)?))
	}
}
