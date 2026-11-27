//! GoEmotions multi-label comment corpus as a device-resident
//! [`oa::data::Dataset`].
//!
//! The `train`/`dev`/`test` TSV files materialize through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `go_emotions/1.0.0`. Each comment encodes as a `[text_len]` F32
//! UTF-8 byte window, truncated or zero-padded; `labels()` is a
//! `[N, 28]` F32 multi-hot matrix over the 27-emotion-plus-neutral
//! taxonomy (upstream column 2 carries comma-separated emotion ids).

use oa::data::{Dataset, Sample};
use oa::matrix::{self, Matrix};
use oa::{Error, Result, Runtime};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::pack::{self, FetchProgress};
use crate::core::tsv;

#[path = "go_emotions.gen.rs"]
mod generated;

pub(crate) use generated::GO_EMOTIONS_PACK;
pub use generated::{EMOTION_COUNT, GoEmotionsConfig, GoEmotionsSplit};

fn member(split: GoEmotionsSplit) -> &'static str {
	match split {
		GoEmotionsSplit::Train => "train.tsv",
		GoEmotionsSplit::Validation => "dev.tsv",
		GoEmotionsSplit::Test => "test.tsv",
	}
}

/// Device-resident GoEmotions split implementing [`oa::data::Dataset`].
///
/// `text()` is `[N, text_len]` F32 byte windows and `labels()` is
/// `[N, 28]` F32 multi-hot emotion indicators.
pub struct GoEmotions {
	text: Matrix,
	labels: Matrix,
	split: GoEmotionsSplit,
	text_len: usize,
}

impl GoEmotions {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, TSV decode errors, and
	/// an error when an emotion id is out of range.
	pub fn decode(config: &GoEmotionsConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled`
	/// when `progress` aborts a fetch.
	pub fn decode_with_progress(
		config: &GoEmotionsConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&GO_EMOTIONS_PACK, &root, progress)?;
		let body = std::fs::read(dir.join(member(config.split)))
			.map_err(|error| Error::io("read go_emotions member", error))?;
		let records = tsv::parse("go_emotions", &body, false)?;
		let scale = if config.normalize { 1.0 / 255.0 } else { 1.0 };
		let mut text = vec![0.0_f32; records.len() * config.text_len];
		let mut labels = vec![0.0_f32; records.len() * EMOTION_COUNT];
		for (row, record) in records.iter().enumerate() {
			if record.len() < 2 {
				continue;
			}
			let bytes = record[0].as_bytes();
			let take = bytes.len().min(config.text_len);
			for (out, &byte) in text[row * config.text_len..row * config.text_len + take]
				.iter_mut()
				.zip(&bytes[..take])
			{
				*out = f32::from(byte) * scale;
			}
			for id in record[1]
				.split(',')
				.map(str::trim)
				.filter(|id| !id.is_empty())
			{
				let index: usize = id.parse().map_err(|_| {
					Error::data_loss(format!(
						"go_emotions row {row} emotion id '{id}' is not a number"
					))
				})?;
				let slot = labels.get_mut(row * EMOTION_COUNT + index).ok_or_else(|| {
					Error::data_loss(format!(
						"go_emotions row {row} emotion id {index} is out of range"
					))
				})?;
				*slot = 1.0;
			}
		}
		Ok(Decoded {
			matrices: vec![
				MatrixData {
					name: "text",
					shape: vec![records.len(), config.text_len],
					values: text,
				},
				MatrixData {
					name: "labels",
					shape: vec![records.len(), EMOTION_COUNT],
					values: labels,
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
		engine: &Runtime,
		config: &GoEmotionsConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Ok(Self {
			text: decoded.upload("text", engine)?,
			labels: decoded.upload("labels", engine)?,
			split: config.split,
			text_len: config.text_len,
		})
	}

	/// Full `[N, text_len]` F32 byte-window matrix on the owning engine.
	pub const fn text(&self) -> &Matrix {
		&self.text
	}

	/// Full `[N, 28]` F32 multi-hot emotion matrix on the owning engine.
	pub const fn labels(&self) -> &Matrix {
		&self.labels
	}

	/// Selected split.
	pub const fn split(&self) -> GoEmotionsSplit {
		self.split
	}

	/// Document window length.
	pub const fn text_len(&self) -> usize {
		self.text_len
	}
}

impl Dataset for GoEmotions {
	fn len(&self) -> i64 {
		self.text.shape()[0] as i64
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		matrix::slice(&self.text, 0, index, index + 1)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		Ok(Sample::new_with_label(
			self.get_item(index)?,
			matrix::slice(&self.labels, 0, index, index + 1)?,
		))
	}
}
