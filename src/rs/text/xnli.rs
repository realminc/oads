//! XNLI cross-lingual NLI dev/test as a TSV pack.
//!
//! `XNLI-1.0.zip` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `xnli/1.0.0`
//! and the selected split member (`xnli.dev.tsv` / `xnli.test.tsv`)
//! decodes to `premise` and `hypothesis` `[N, K_i]` F32 byte matrices
//! (columns 7 and 8 of the official header), a `[N]` F32 label
//! matrix, and a `[N]` F32 language-id matrix indexing
//! [`LANGUAGES`]. `get_item` returns the `[1, K1 + K2]` end-to-end
//! premise+hypothesis row.

use oa::data::{Dataset, Sample};
use oa::matrix::{self, Matrix};
use oa::{Error, Result, Runtime};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::pack::{self, FetchProgress};
use crate::core::{extract, tsv};

#[path = "xnli.gen.rs"]
mod generated;

pub(crate) use generated::XNLI_PACK;
pub use generated::{XnliConfig, XnliSplit};

/// ISO 639-1 codes in `languages()` matrix index order — the 15
/// evaluation languages of the official XNLI dev/test members.
pub const LANGUAGES: &[&str] = &[
	"ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh",
];

const LABELS: &[&str] = &["entailment", "neutral", "contradiction"];
/// Display names; label index `i` names `CLASSES[i]`.
pub const CLASSES: &[&str] = &["entailment", "neutral", "contradiction"];

const LANGUAGE_COL: usize = 0;
const LABEL_COL: usize = 1;
const TEXT_COLS: [usize; 2] = [6, 7];
const FIELDS: usize = 8;

fn member(split: XnliSplit) -> &'static str {
	match split {
		XnliSplit::Validation => "xnli.dev.tsv",
		XnliSplit::Test => "xnli.test.tsv",
	}
}

fn decode_rows(pack: &str, rows: &[Vec<String>]) -> Result<Decoded> {
	let mut widths = [0_usize; 2];
	for (row, record) in rows.iter().enumerate() {
		if record.len() < FIELDS {
			return Err(Error::data_loss(format!(
				"{pack} tsv row {row} has {} fields, expected at least {FIELDS}",
				record.len()
			)));
		}
		for (field, &col) in TEXT_COLS.iter().enumerate() {
			widths[field] = widths[field].max(record[col].len());
		}
	}
	let mut matrices = Vec::with_capacity(4);
	for (field, &col) in TEXT_COLS.iter().enumerate() {
		let mut values = vec![0.0f32; rows.len() * widths[field]];
		for (row, record) in rows.iter().enumerate() {
			for (k, &byte) in record[col].as_bytes().iter().enumerate() {
				values[row * widths[field] + k] = f32::from(byte);
			}
		}
		matrices.push(MatrixData {
			name: if field == 0 { "premise" } else { "hypothesis" },
			shape: vec![rows.len(), widths[field]],
			values,
		});
	}
	let mut labels = Vec::with_capacity(rows.len());
	let mut languages = Vec::with_capacity(rows.len());
	for (row, record) in rows.iter().enumerate() {
		labels.push(
			LABELS
				.iter()
				.position(|name| *name == record[LABEL_COL])
				.map(|index| index as f32)
				.ok_or_else(|| {
					Error::data_loss(format!(
						"{pack} tsv row {row} label '{}' is unknown",
						record[LABEL_COL]
					))
				})?,
		);
		languages.push(
			LANGUAGES
				.iter()
				.position(|name| *name == record[LANGUAGE_COL])
				.map(|index| index as f32)
				.ok_or_else(|| {
					Error::data_loss(format!(
						"{pack} tsv row {row} language '{}' is unknown",
						record[LANGUAGE_COL]
					))
				})?,
		);
	}
	matrices.push(MatrixData {
		name: "labels",
		shape: vec![labels.len()],
		values: labels,
	});
	matrices.push(MatrixData {
		name: "languages",
		shape: vec![languages.len()],
		values: languages,
	});
	Ok(Decoded {
		matrices,
		text: None,
		seq_len: 0,
	})
}

fn decode_pack(pack_dir: &std::path::Path, member_name: &str) -> Result<Decoded> {
	let suffix = format!("/{member_name}");
	let mut body: Option<Vec<u8>> = None;
	extract::walk_zip_file(
		&pack_dir.join("XNLI-1.0.zip"),
		|name| name.ends_with(&suffix),
		|name, data| {
			if name.ends_with(&suffix) {
				body = Some(data.to_vec());
			}
			Ok(())
		},
	)?;
	let body =
		body.ok_or_else(|| Error::data_loss(format!("xnli archive lacks member {member_name}")))?;
	let rows = tsv::parse("xnli", &body, true)?;
	decode_rows("xnli", &rows)
}

/// XNLI: 15-language premise/hypothesis pairs with NLI labels and
/// per-row language ids.
pub struct Xnli {
	premise: Matrix,
	hypothesis: Matrix,
	labels: Matrix,
	languages: Matrix,
}

impl Xnli {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and TSV decode errors.
	pub fn decode(config: &XnliConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus
	/// `Cancelled` when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &XnliConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&XNLI_PACK, &root, progress)?;
		decode_pack(&dir, member(config.split))
	}

	/// [`load_on`](Self::load_on) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`load_on`](Self::load_on), plus
	/// `Cancelled` when the callback aborts a fetch.
	pub fn load_with_progress_on(
		engine: &Runtime,
		config: &XnliConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Ok(Self {
			premise: decoded.upload("premise", engine)?,
			hypothesis: decoded.upload("hypothesis", engine)?,
			labels: decoded.upload("labels", engine)?,
			languages: decoded.upload("languages", engine)?,
		})
	}

	/// `[N, K]` F32 premise byte matrix.
	#[must_use]
	pub const fn premise(&self) -> &Matrix {
		&self.premise
	}

	/// `[N, K]` F32 hypothesis byte matrix.
	#[must_use]
	pub const fn hypothesis(&self) -> &Matrix {
		&self.hypothesis
	}

	/// `[N]` F32 label matrix; index `i` names [`CLASSES`]`[i]`.
	#[must_use]
	pub const fn labels(&self) -> &Matrix {
		&self.labels
	}

	/// `[N]` F32 language-id matrix; index `i` names
	/// [`LANGUAGES`]`[i]`.
	#[must_use]
	pub const fn languages(&self) -> &Matrix {
		&self.languages
	}

	/// Class display names; label index `i` names `classes()[i]`.
	#[must_use]
	pub fn classes(&self) -> &[&'static str] {
		CLASSES
	}
}

impl Dataset for Xnli {
	fn len(&self) -> i64 {
		i64::try_from(self.premise.shape()[0]).unwrap_or(i64::MAX)
	}

	/// `[1, K1 + K2]` end-to-end premise+hypothesis byte row.
	fn get_item(&self, index: i64) -> Result<Matrix> {
		matrix::concat(
			&[
				matrix::slice(&self.premise, 0, index, index + 1)?,
				matrix::slice(&self.hypothesis, 0, index, index + 1)?,
			],
			1,
		)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		Ok(Sample::new_with_label(
			self.get_item(index)?,
			matrix::slice(&self.labels, 0, index, index + 1)?,
		))
	}
}
