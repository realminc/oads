//! SQuAD 1.1 reading-comprehension corpus as a
//! [`oa::data::Dataset`].
//!
//! `{train,dev}-v1.1.json` are materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `squad_v1/1.0.0`. Text columns are variable-length, so each decodes
//! to a flat `[B]` F32 byte blob plus `[N]`/`[P]` `offsets` and
//! `lengths` tables; contexts additionally deduplicate to `P` unique
//! paragraphs indexed by `[N]` `context_index`. `answer_start` is
//! `[N]` F32. `get_item` concatenates
//! one question and context slice to `[Kq_i + Kc_i]`.

use oa::data::{Dataset, Sample};
use oa::matrix::{self, Matrix};
use oa::{Engine, Error, Result};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::pack::{self, FetchProgress};

#[path = "squad_v1.gen.rs"]
mod generated;

pub(crate) use generated::SQUAD_V1_PACK;
pub use generated::{SquadV1Config, SquadV1Split};

fn member(split: SquadV1Split) -> &'static str {
	match split {
		SquadV1Split::Train => "train-v1.1.json",
		SquadV1Split::Validation => "dev-v1.1.json",
	}
}

/// One flattened QA row.
struct Qa {
	question: String,
	answer: String,
	answer_start: f32,
}

/// Materialized SQuAD 1.1 split on an engine.
pub struct SquadV1 {
	question_bytes: Matrix,
	question_offsets: Matrix,
	question_lengths: Matrix,
	context_bytes: Matrix,
	context_offsets: Matrix,
	context_lengths: Matrix,
	context_index: Matrix,
	answer_bytes: Matrix,
	answer_offsets: Matrix,
	answer_lengths: Matrix,
	answer_start: Matrix,
	question_span: Vec<(u32, u32)>,
	context_span: Vec<(u32, u32)>,
	context_lookup: Vec<u32>,
	split: SquadV1Split,
}

impl SquadV1 {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and JSON decode errors.
	pub fn decode(config: &SquadV1Config) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled`
	/// when the callback aborts a fetch.
	pub fn decode_with_progress(
		config: &SquadV1Config,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&SQUAD_V1_PACK, &root, progress)?;
		let body = std::fs::read(dir.join(member(config.split)))
			.map_err(|error| Error::io("read squad_v1 member", error))?;
		let (qas, contexts) = flatten(&body)?;
		Ok(decoded(qas, contexts))
	}

	/// [`load_on`](Self::load_on) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`load_on`](Self::load_on), plus `Cancelled`
	/// when the callback aborts a fetch.
	pub fn load_with_progress_on(
		engine: &Engine,
		config: &SquadV1Config,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		let spans = |name: &str| -> Vec<(u32, u32)> {
			let (starts, lens) = (format!("{name}_offsets"), format!("{name}_lengths"));
			let find = |key: &str| -> Vec<u32> {
				decoded
					.matrices
					.iter()
					.find(|m| m.name == key)
					.map(|m| m.values.iter().map(|&v| v as u32).collect())
					.unwrap_or_default()
			};
			find(&starts).into_iter().zip(find(&lens)).collect()
		};
		let context_lookup = decoded
			.matrices
			.iter()
			.find(|m| m.name == "context_index")
			.map(|m| m.values.iter().map(|&v| v as u32).collect::<Vec<_>>())
			.unwrap_or_default();
		Ok(Self {
			question_span: spans("question"),
			context_span: spans("context"),
			context_lookup,
			question_bytes: decoded.upload("question_bytes", engine)?,
			question_offsets: decoded.upload("question_offsets", engine)?,
			question_lengths: decoded.upload("question_lengths", engine)?,
			context_bytes: decoded.upload("context_bytes", engine)?,
			context_offsets: decoded.upload("context_offsets", engine)?,
			context_lengths: decoded.upload("context_lengths", engine)?,
			context_index: decoded.upload("context_index", engine)?,
			answer_bytes: decoded.upload("answer_bytes", engine)?,
			answer_offsets: decoded.upload("answer_offsets", engine)?,
			answer_lengths: decoded.upload("answer_lengths", engine)?,
			answer_start: decoded.upload("answer_start", engine)?,
			split: config.split,
		})
	}

	/// `[Bq]` F32 flat question byte blob on the owning engine;
	/// `question_offsets`/`question_lengths` carve one span per row.
	#[must_use]
	pub const fn question_bytes(&self) -> &Matrix {
		&self.question_bytes
	}

	/// `[N]` F32 byte offsets of each question inside `question_bytes`.
	#[must_use]
	pub const fn question_offsets(&self) -> &Matrix {
		&self.question_offsets
	}

	/// `[N]` F32 byte lengths of each question.
	#[must_use]
	pub const fn question_lengths(&self) -> &Matrix {
		&self.question_lengths
	}

	/// `[Bc]` F32 flat deduplicated context byte blob on the owning
	/// engine; `context_offsets`/`context_lengths` carve one span per
	/// paragraph and `context_index` selects one per question.
	#[must_use]
	pub const fn context_bytes(&self) -> &Matrix {
		&self.context_bytes
	}

	/// `[P]` F32 byte offsets of each paragraph inside `context_bytes`.
	#[must_use]
	pub const fn context_offsets(&self) -> &Matrix {
		&self.context_offsets
	}

	/// `[P]` F32 byte lengths of each paragraph.
	#[must_use]
	pub const fn context_lengths(&self) -> &Matrix {
		&self.context_lengths
	}

	/// `[N]` F32 row indices into the deduplicated context table, one
	/// per question.
	#[must_use]
	pub const fn context_index(&self) -> &Matrix {
		&self.context_index
	}

	/// `[Ba]` F32 flat answer-text byte blob.
	#[must_use]
	pub const fn answer_bytes(&self) -> &Matrix {
		&self.answer_bytes
	}

	/// `[N]` F32 byte offsets of each answer inside `answer_bytes`.
	#[must_use]
	pub const fn answer_offsets(&self) -> &Matrix {
		&self.answer_offsets
	}

	/// `[N]` F32 byte lengths of each answer.
	#[must_use]
	pub const fn answer_lengths(&self) -> &Matrix {
		&self.answer_lengths
	}

	/// `[N]` F32 answer start offsets in the context byte string.
	#[must_use]
	pub const fn answer_start(&self) -> &Matrix {
		&self.answer_start
	}

	/// The split this instance loaded.
	#[must_use]
	pub const fn split(&self) -> SquadV1Split {
		self.split
	}
}

impl Dataset for SquadV1 {
	fn len(&self) -> i64 {
		i64::from(self.question_offsets.shape()[0] as i32)
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		let i = usize::try_from(index)
			.map_err(|_| Error::invalid_argument("squad_v1 index must be nonnegative"))?;
		let (qs, ql) = *self
			.question_span
			.get(i)
			.ok_or_else(|| Error::invalid_argument("squad_v1 index is out of bounds"))?;
		let row = *self
			.context_lookup
			.get(i)
			.ok_or_else(|| Error::invalid_argument("squad_v1 index is out of bounds"))?;
		let (cs, cl) = self.context_span[row as usize];
		let q = matrix::slice(&self.question_bytes, 0, i64::from(qs), i64::from(qs + ql))?;
		let c = matrix::slice(&self.context_bytes, 0, i64::from(cs), i64::from(cs + cl))?;
		matrix::concat(&[q, c], 0)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		let x = self.get_item(index)?;
		let y = matrix::slice(&self.answer_start, 0, index, index + 1)?;
		Ok(Sample::new_with_label(x, y))
	}
}

/// One flattened QA row plus its shared-context index.
struct QaRow {
	qa: Qa,
	context_index: u32,
}

/// Flatten `paragraphs[].qas[]` into per-question rows and the
/// deduplicated context table (first-seen order).
fn flatten(body: &[u8]) -> Result<(Vec<QaRow>, Vec<String>)> {
	let root: serde_json::Value = serde_json::from_slice(body)
		.map_err(|error| Error::data_loss(format!("squad_v1 member is not valid JSON: {error}")))?;
	let data = root["data"]
		.as_array()
		.ok_or_else(|| Error::data_loss("squad_v1 lacks data[]"))?;
	let mut qas: Vec<QaRow> = Vec::new();
	let mut contexts: Vec<String> = Vec::new();
	let mut context_ids: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
	for article in data {
		let paragraphs = article["paragraphs"]
			.as_array()
			.ok_or_else(|| Error::data_loss("squad_v1 article lacks paragraphs[]"))?;
		for paragraph in paragraphs {
			let context = paragraph["context"].as_str().unwrap_or("");
			let next = u32::try_from(contexts.len())
				.map_err(|_| Error::data_loss("squad_v1 context table overflow"))?;
			let context_index = *context_ids.entry(context).or_insert_with(|| {
				contexts.push(context.to_owned());
				next
			});
			for qa in paragraph["qas"].as_array().unwrap_or(&vec![]) {
				let (answer, start) = qa["answers"]
					.as_array()
					.and_then(|answers| answers.first())
					.map(|first| {
						(
							first["text"].as_str().unwrap_or("").to_owned(),
							first["answer_start"].as_f64().unwrap_or(-1.0) as f32,
						)
					})
					.unwrap_or((String::new(), -1.0));
				qas.push(QaRow {
					qa: Qa {
						question: qa["question"].as_str().unwrap_or("").to_owned(),
						answer,
						answer_start: start,
					},
					context_index,
				});
			}
		}
	}
	Ok((qas, contexts))
}

/// Flatten a string column to a `[B]` byte blob plus `[N]` offsets and
/// lengths.
fn blob(strings: impl Iterator<Item = String>) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
	let mut bytes = Vec::new();
	let mut offsets = Vec::new();
	let mut lengths = Vec::new();
	for text in strings {
		offsets.push(bytes.len() as f32);
		lengths.push(text.len() as f32);
		bytes.extend(text.bytes().map(f32::from));
	}
	(bytes, offsets, lengths)
}

/// Build the matrix payload for `qas` over the deduplicated `contexts`.
fn decoded(qas: Vec<QaRow>, contexts: Vec<String>) -> Decoded {
	let (q_bytes, q_off, q_len) = blob(qas.iter().map(|row| row.qa.question.clone()));
	let (c_bytes, c_off, c_len) = blob(contexts.into_iter());
	let (a_bytes, a_off, a_len) = blob(qas.iter().map(|row| row.qa.answer.clone()));
	let n = qas.len();
	let p = c_off.len();
	Decoded {
		matrices: vec![
			MatrixData {
				name: "question_bytes",
				shape: vec![q_bytes.len()],
				values: q_bytes,
			},
			MatrixData {
				name: "question_offsets",
				shape: vec![n],
				values: q_off,
			},
			MatrixData {
				name: "question_lengths",
				shape: vec![n],
				values: q_len,
			},
			MatrixData {
				name: "context_bytes",
				shape: vec![c_bytes.len()],
				values: c_bytes,
			},
			MatrixData {
				name: "context_offsets",
				shape: vec![p],
				values: c_off,
			},
			MatrixData {
				name: "context_lengths",
				shape: vec![p],
				values: c_len,
			},
			MatrixData {
				name: "context_index",
				shape: vec![n],
				values: qas.iter().map(|row| row.context_index as f32).collect(),
			},
			MatrixData {
				name: "answer_bytes",
				shape: vec![a_bytes.len()],
				values: a_bytes,
			},
			MatrixData {
				name: "answer_offsets",
				shape: vec![n],
				values: a_off,
			},
			MatrixData {
				name: "answer_lengths",
				shape: vec![n],
				values: a_len,
			},
			MatrixData {
				name: "answer_start",
				shape: vec![n],
				values: qas.iter().map(|row| row.qa.answer_start).collect(),
			},
		],
		text: None,
		seq_len: 0,
	}
}
