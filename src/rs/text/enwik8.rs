//! enwik8 (Hutter Prize) byte corpus as a device-resident
//! [`oa::data::Dataset`].
//!
//! `enwik8.zip` materializes through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `enwik8/1.0.0`
//! and its single `enwik8` member carries the first 100,000,000 bytes
//! of a Wikipedia XML dump. The conventional byte-range split applies:
//! bytes `[0, 90M)` train, `[90M, 95M)` validation, `[95M, 100M)` test.
//! A sample is a `seq_len`-byte token window whose label is the same
//! window shifted one byte ahead — the standard character
//! language-model contract.

use oa::data::{Dataset, Sample};
use oa::matrix::{self, Matrix};
use oa::{Engine, Error, Result};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::extract;
use crate::core::pack::{self, FetchProgress};

#[path = "enwik8.gen.rs"]
mod generated;

pub(crate) use generated::ENWIK8_PACK;
pub use generated::{Enwik8Config, Enwik8Split};

fn range(split: Enwik8Split) -> (usize, usize) {
	match split {
		Enwik8Split::Train => (0, 90_000_000),
		Enwik8Split::Validation => (90_000_000, 95_000_000),
		Enwik8Split::Test => (95_000_000, 100_000_000),
	}
}

/// Device-resident enwik8 byte range implementing [`oa::data::Dataset`].
///
/// `tokens()` is `[N]` F32 byte values `0..=255` for the selected
/// range. `len()` is `N - seq_len`; `get_item` returns the
/// `[seq_len]` window `tokens[i..i + seq_len]` and `get_sample` pairs
/// it with the label window shifted one byte ahead.
pub struct Enwik8 {
	tokens: Matrix,
	split: Enwik8Split,
	seq_len: usize,
}

impl Enwik8 {
	/// Fetch, verify, and decode the selected byte range on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, zip member errors, and
	/// an error when the archive is shorter than the selected range.
	pub fn decode(config: &Enwik8Config) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled`
	/// when `progress` aborts a fetch.
	pub fn decode_with_progress(
		config: &Enwik8Config,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&ENWIK8_PACK, &root, progress)?;
		let mut body: Option<Vec<u8>> = None;
		extract::walk_zip_file(
			&dir.join("enwik8.zip"),
			|name| name == "enwik8",
			|name, data| {
				if name == "enwik8" {
					body = Some(data.to_vec());
				}
				Ok(())
			},
		)?;
		let body = body.ok_or_else(|| Error::data_loss("enwik8 archive lacks enwik8"))?;
		let (start, end) = range(config.split);
		let bytes = body
			.get(start..end)
			.ok_or_else(|| Error::data_loss("enwik8 archive is shorter than the selected range"))?;
		let values = bytes
			.iter()
			.map(|&byte| f32::from(byte))
			.collect::<Vec<_>>();
		Ok(Decoded {
			matrices: vec![MatrixData {
				name: "tokens",
				shape: vec![values.len()],
				values,
			}],
			text: None,
			seq_len: 0,
		})
	}

	/// Decode the range and upload its tokens to `engine`, with a fetch
	/// progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled`
	/// when `progress` aborts a fetch, an error when `seq_len` is 0 or
	/// leaves no sample window, and matrix upload errors.
	pub fn load_with_progress_on(
		engine: &Engine,
		config: &Enwik8Config,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		if config.seq_len == 0 {
			return Err(Error::invalid_argument("enwik8 seq_len must be at least 1"));
		}
		let decoded = Self::decode_with_progress(config, progress)?;
		let count = decoded.matrix("tokens")?.shape[0];
		if count <= config.seq_len {
			return Err(Error::invalid_argument(format!(
				"enwik8 seq_len {} leaves no sample in {count} bytes",
				config.seq_len,
			)));
		}
		Ok(Self {
			tokens: decoded.upload("tokens", engine)?,
			split: config.split,
			seq_len: config.seq_len,
		})
	}

	/// Full `[N]` F32 token matrix (byte values `0..=255`) on the engine.
	#[must_use]
	pub const fn tokens(&self) -> &Matrix {
		&self.tokens
	}

	/// Selected byte-range split.
	#[must_use]
	pub const fn split(&self) -> Enwik8Split {
		self.split
	}

	/// The configured token window length.
	#[must_use]
	pub const fn seq_len(&self) -> usize {
		self.seq_len
	}

	fn check_index(&self, index: i64) -> Result<()> {
		if index < 0 || index >= self.len() {
			return Err(Error::out_of_range("enwik8 index is outside the dataset"));
		}
		Ok(())
	}
}

impl Dataset for Enwik8 {
	fn len(&self) -> i64 {
		i64::try_from(self.tokens.shape()[0].saturating_sub(self.seq_len)).unwrap_or(i64::MAX)
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		self.check_index(index)?;
		matrix::slice(&self.tokens, 0, index, index + self.seq_len as i64)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		self.check_index(index)?;
		let end = index + self.seq_len as i64;
		let x = matrix::slice(&self.tokens, 0, index, end)?;
		let y = matrix::slice(&self.tokens, 0, index + 1, end + 1)?;
		Ok(Sample::new_with_label(x, y))
	}
}
