//! Multi-Sentence Reading Comprehension answer-validation task as a SuperGLUE JSONL pack.
//!
//! `MultiRC.zip` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `superglue_multirc/1.0.0`
//! and the selected split member decodes through `recordl`: one
//! `[N, K_i]` F32 byte matrix per text field plus the `[N]` F32 label matrix.
//! `get_item` returns the `[1, sum(K_i)]` end-to-end field row.

use oa::data::{Dataset, Sample};
use oa::matrix::Matrix;
use oa::{Result, Runtime};

use crate::core::decode::Decoded;
use crate::core::jsonl;
use crate::core::pack::FetchProgress;
use crate::text::recordl::{self, RecordData, RecordSpec};

#[path = "superglue_multirc.gen.rs"]
mod generated;

pub(crate) use generated::SUPERGLUE_MULTIRC_PACK;
pub use generated::{SuperglueMultircConfig, SuperglueMultircSplit};

fn map(pack: &str, value: &serde_json::Value, row: usize) -> Result<Vec<(Vec<String>, String)>> {
	let passage = jsonl::scalar_at(pack, value, "passage.text", row)?;
	let questions = value
		.get("passage")
		.and_then(|node| node.get("questions"))
		.and_then(serde_json::Value::as_array)
		.ok_or_else(|| {
			oa::Error::data_loss(format!("{pack} jsonl row {row} lacks 'passage.questions'"))
		})?;
	let mut rows = Vec::new();
	for question in questions {
		let text = jsonl::scalar_at_opt(question, "question");
		let answers = question
			.get("answers")
			.and_then(serde_json::Value::as_array)
			.ok_or_else(|| oa::Error::data_loss(format!("{pack} jsonl row {row} lacks 'answers'")))?;
		for answer in answers {
			rows.push((
				vec![
					passage.clone(),
					text.clone(),
					jsonl::scalar_at_opt(answer, "text"),
				],
				jsonl::scalar_at_opt(answer, "label"),
			));
		}
	}
	Ok(rows)
}

const SPEC: RecordSpec = RecordSpec {
	fields: &["passage", "question", "answer"],
	label_names: &[],
	classes: &["incorrect", "correct"],
	map,
	unlabeled: &["test.jsonl"],
};

fn member(split: SuperglueMultircSplit) -> &'static str {
	match split {
		SuperglueMultircSplit::Train => "train.jsonl",
		SuperglueMultircSplit::Validation => "val.jsonl",
		SuperglueMultircSplit::Test => "test.jsonl",
	}
}

/// SuperglueMultirc JSONL record dataset over 3 text field(s).
pub struct SuperglueMultirc {
	data: RecordData,
}

impl SuperglueMultirc {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and JSONL decode errors.
	pub fn decode(config: &SuperglueMultircConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus
	/// `Cancelled` when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &SuperglueMultircConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => crate::core::pack::data_root()?,
		};
		let dir =
			crate::core::pack::ensure_pack_with_progress(&SUPERGLUE_MULTIRC_PACK, &root, progress)?;
		let (decoded, _) = recordl::decode_member(
			"superglue_multirc",
			&dir.join("MultiRC.zip"),
			member(config.split),
			&SPEC,
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
		engine: &Runtime,
		config: &SuperglueMultircConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let data = RecordData::load_on(
			engine,
			&SUPERGLUE_MULTIRC_PACK,
			std::path::Path::new("MultiRC.zip"),
			member(config.split),
			&SPEC,
			config.cache_dir.clone(),
			progress,
		)?;
		Ok(Self { data })
	}

	/// `[N, K]` F32 byte matrix for text field `field` (0-based, in
	/// the documented field order).
	///
	/// # Errors
	///
	/// Returns `invalid_argument` when `field` is out of range.
	pub fn field(&self, field: usize) -> Result<&Matrix> {
		self
			.data
			.fields
			.get(field)
			.ok_or_else(|| oa::Error::invalid_argument("superglue_multirc field index out of range"))
	}

	/// `[N]` F32 label matrix; `None` for unlabeled members.
	#[must_use]
	pub const fn labels(&self) -> Option<&Matrix> {
		self.data.labels.as_ref()
	}

	/// Class display names; label index `i` names `classes()[i]`.
	#[must_use]
	pub fn classes(&self) -> &[String] {
		&self.data.classes
	}
}

impl Dataset for SuperglueMultirc {
	fn len(&self) -> i64 {
		i64::try_from(self.data.rows()).unwrap_or(i64::MAX)
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		self.data.item(index)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		self.data.sample(index)
	}
}
