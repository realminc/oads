//! Tiny Shakespeare as a device-resident character dataset.
//!
//! The ~1.1 MB `input.txt` is materialized through [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `tiny_shakespeare/1.0.0`, decoded to UTF-8 text on the host, and uploaded
//! once as a `[N]` F32 matrix of byte values `0..=255`. A sample is a
//! `seq_len`-byte token window whose label is the same window shifted one
//! byte ahead — the standard character language-model contract.

use oa::data::{Dataset, Sample};
use oa::matrix::{self, Matrix};
use oa::{Error, Result, Runtime};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::pack::{self, FetchProgress};

#[path = "tiny_shakespeare.gen.rs"]
mod generated;

pub(crate) use generated::SHAKESPEARE_PACK;
pub use generated::TinyShakespeareConfig;

/// Device-resident Tiny Shakespeare text implementing [`oa::data::Dataset`].
///
/// `tokens()` is `[N]` F32 byte values `0..=255` on the owning engine.
/// `len()` is `N - seq_len`; `get_item` returns the `[seq_len]` window
/// `tokens[i..i + seq_len]` and `get_sample` pairs it with the label window
/// `tokens[i + 1..i + seq_len + 1]`.
pub struct TinyShakespeare {
	tokens: Matrix,
	text: String,
	seq_len: usize,
}

impl TinyShakespeare {
	/// Fetch, verify, and decode the corpus on the host.
	///
	/// Returns the full corpus as `[N]` F32 byte tokens plus the decoded
	/// UTF-8 text; `config.seq_len` is a dataset-level option that does not
	/// affect the decode.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and an error when the payload
	/// is not valid UTF-8.
	pub fn decode(config: &TinyShakespeareConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// the callback aborts a fetch.
	pub fn decode_with_progress(
		config: &TinyShakespeareConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&SHAKESPEARE_PACK, &root, progress)?;
		let text = std::fs::read_to_string(dir.join("input.txt"))
			.map_err(|error| Error::io("read tiny_shakespeare text", error))?;
		let values = text.bytes().map(f32::from).collect::<Vec<_>>();
		let count = values.len();
		Ok(Decoded {
			matrices: vec![MatrixData {
				name: "tokens",
				shape: vec![count],
				values,
			}],
			text: Some(text),
			seq_len: 0,
		})
	}

	/// Decode the corpus and upload its tokens to `engine`, with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, an error when `seq_len` is 0
	/// or leaves no sample window, an error when the payload is not valid
	/// UTF-8, and matrix upload errors.
	pub fn load_with_progress_on(
		engine: &Runtime,
		config: &TinyShakespeareConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		if config.seq_len == 0 {
			return Err(Error::invalid_argument(
				"tiny_shakespeare seq_len must be at least 1",
			));
		}
		let decoded = Self::decode_with_progress(config, progress)?;
		let count = decoded.matrix("tokens")?.shape[0];
		if count <= config.seq_len {
			return Err(Error::invalid_argument(format!(
				"tiny_shakespeare seq_len {} leaves no sample in {count} bytes",
				config.seq_len,
			)));
		}
		Ok(Self {
			tokens: decoded.upload("tokens", engine)?,
			text: decoded.text.unwrap_or_default(),
			seq_len: config.seq_len,
		})
	}

	/// Full `[N]` F32 token matrix (byte values `0..=255`) on the engine.
	#[must_use]
	pub const fn tokens(&self) -> &Matrix {
		&self.tokens
	}

	/// The decoded UTF-8 text held on the host.
	#[must_use]
	pub fn text(&self) -> &str {
		&self.text
	}

	/// The configured token window length.
	#[must_use]
	pub const fn seq_len(&self) -> usize {
		self.seq_len
	}

	fn check_index(&self, index: i64) -> Result<()> {
		if index < 0 || index >= self.len() {
			return Err(Error::out_of_range(
				"tiny_shakespeare index is outside the dataset",
			));
		}
		Ok(())
	}
}

impl Dataset for TinyShakespeare {
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
