//! Reading Comprehension with Commonsense Reasoning cloze task as a SuperGLUE JSONL pack.
//!
//! `ReCoRD.zip` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `superglue_record/1.0.0`
//! and the selected split member decodes through `recordl`: one
//! `[N, K_i]` F32 byte matrix per text field and no label matrix (the answer is a text field).
//! `get_item` returns the `[1, sum(K_i)]` end-to-end field row.

use oa::data::{Dataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Result};

use crate::core::decode::Decoded;
use crate::core::jsonl;
use crate::core::pack::FetchProgress;
use crate::text::recordl::{self, RecordData, RecordSpec};

#[path = "superglue_record.gen.rs"]
mod generated;

pub(crate) use generated::SUPERGLUE_RECORD_PACK;
pub use generated::{SuperglueRecordConfig, SuperglueRecordSplit};

fn map(pack: &str, value: &serde_json::Value, row: usize) -> Result<Vec<(Vec<String>, String)>> {
	{
		let passage = jsonl::scalar_at(pack, value, "passage.text", row)?;
		let qas = value
			.get("qas")
			.and_then(serde_json::Value::as_array)
			.ok_or_else(|| oa::Error::data_loss(format!("{pack} jsonl row {row} lacks 'qas' array")))?;
		let mut rows = Vec::with_capacity(qas.len());
		for qa in qas {
			let query = jsonl::scalar_at_opt(qa, "query");
			let answer = qa
				.get("answers")
				.and_then(serde_json::Value::as_array)
				.and_then(|answers| answers.first())
				.map(|first| jsonl::scalar_at_opt(first, "text"))
				.unwrap_or_default();
			rows.push((vec![passage.clone(), query, answer], String::new()));
		}
		Ok(rows)
	}
}

const SPEC: RecordSpec = RecordSpec {
	fields: &["passage", "query", "answer"],
	label_names: &[],
	classes: &[],
	map,
	unlabeled: &[".jsonl"],
};

fn member(split: SuperglueRecordSplit) -> &'static str {
	match split {
		SuperglueRecordSplit::Train => "train.jsonl",
		SuperglueRecordSplit::Validation => "val.jsonl",
		SuperglueRecordSplit::Test => "test.jsonl",
	}
}

/// SuperglueRecord JSONL record dataset over 3 text field(s).
pub struct SuperglueRecord {
	data: RecordData,
}

impl SuperglueRecord {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and JSONL decode errors.
	pub fn decode(config: &SuperglueRecordConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus
	/// `Cancelled` when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &SuperglueRecordConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => crate::core::pack::data_root()?,
		};
		let dir =
			crate::core::pack::ensure_pack_with_progress(&SUPERGLUE_RECORD_PACK, &root, progress)?;
		let (decoded, _) = recordl::decode_member(
			"superglue_record",
			&dir.join("ReCoRD.zip"),
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
		engine: &Engine,
		config: &SuperglueRecordConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let data = RecordData::load_on(
			engine,
			&SUPERGLUE_RECORD_PACK,
			std::path::Path::new("ReCoRD.zip"),
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
			.ok_or_else(|| oa::Error::invalid_argument("superglue_record field index out of range"))
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

impl Dataset for SuperglueRecord {
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
