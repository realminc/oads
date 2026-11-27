//! ARC-Easy grade-school science multiple-choice questions as a JSONL pack.
//!
//! `ARC-V1-Feb2018.zip` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `arc_easy/1.0.0`
//! and the selected split member decodes through `recordl`: one
//! `[N, K_i]` F32 byte matrix per text field plus the `[N]` F32 label
//! matrix. `get_item` returns the `[1, sum(K_i)]` end-to-end field row.

use oa::data::{Dataset, Sample};
use oa::matrix::Matrix;
use oa::{Result, Runtime};

use crate::core::decode::Decoded;
use crate::core::jsonl;
use crate::core::pack::FetchProgress;
use crate::text::recordl::{self, RecordData, RecordSpec};

#[path = "arc_easy.gen.rs"]
mod generated;

pub(crate) use generated::ARC_EASY_PACK;
pub use generated::{ArcEasyConfig, ArcEasySplit};

fn map(pack: &str, value: &serde_json::Value, row: usize) -> Result<Vec<(Vec<String>, String)>> {
	let choices = value
		.pointer("/question/choices")
		.and_then(|value| value.as_array())
		.ok_or_else(|| oa::Error::data_loss(format!("{pack} jsonl row {row} lacks choices")))?;
	let mut text = String::new();
	for choice in choices {
		if !text.is_empty() {
			text.push('\n');
		}
		text.push_str(choice.get("label").and_then(|v| v.as_str()).unwrap_or(""));
		text.push('\t');
		text.push_str(choice.get("text").and_then(|v| v.as_str()).unwrap_or(""));
	}
	Ok(vec![(
		vec![jsonl::scalar_at(pack, value, "question.stem", row)?, text],
		jsonl::scalar_at(pack, value, "answerKey", row)?,
	)])
}

const SPEC: RecordSpec = RecordSpec {
	fields: &["question", "choices"],
	label_names: &["A", "B", "C", "D", "E", "1", "2", "3", "4", "5"],
	classes: &["A", "B", "C", "D", "E", "1", "2", "3", "4", "5"],
	map,
	unlabeled: &[],
};

fn member(split: ArcEasySplit) -> &'static str {
	match split {
		ArcEasySplit::Train => "ARC-Easy-Train.jsonl",
		ArcEasySplit::Validation => "ARC-Easy-Dev.jsonl",
		ArcEasySplit::Test => "ARC-Easy-Test.jsonl",
	}
}

/// ArcEasy JSONL record dataset over 2 text field(s).
pub struct ArcEasy {
	data: RecordData,
}

impl ArcEasy {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and JSONL decode errors.
	pub fn decode(config: &ArcEasyConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus
	/// `Cancelled` when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &ArcEasyConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => crate::core::pack::data_root()?,
		};
		let dir = crate::core::pack::ensure_pack_with_progress(&ARC_EASY_PACK, &root, progress)?;
		let (decoded, _) = recordl::decode_member(
			"arc_easy",
			&dir.join("ARC-V1-Feb2018.zip"),
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
		config: &ArcEasyConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let data = RecordData::load_on(
			engine,
			&ARC_EASY_PACK,
			std::path::Path::new("ARC-V1-Feb2018.zip"),
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
			.ok_or_else(|| oa::Error::invalid_argument("arc_easy field index out of range"))
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

impl Dataset for ArcEasy {
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
