//! Choice of Plausible Alternatives causal-reasoning task as a SuperGLUE JSONL pack.
//!
//! `COPA.zip` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `superglue_copa/1.0.0`
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

#[path = "superglue_copa.gen.rs"]
mod generated;

pub(crate) use generated::SUPERGLUE_COPA_PACK;
pub use generated::{SuperglueCopaConfig, SuperglueCopaSplit};

fn map(pack: &str, value: &serde_json::Value, row: usize) -> Result<Vec<(Vec<String>, String)>> {
	Ok(vec![(
		vec![
			jsonl::scalar_at(pack, value, "premise", row)?,
			jsonl::scalar_at(pack, value, "choice1", row)?,
			jsonl::scalar_at(pack, value, "choice2", row)?,
			jsonl::scalar_at(pack, value, "question", row)?,
		],
		jsonl::scalar_at_opt(value, "label"),
	)])
}

const SPEC: RecordSpec = RecordSpec {
	fields: &["premise", "choice1", "choice2", "question"],
	label_names: &[],
	classes: &["choice1", "choice2"],
	map,
	unlabeled: &["test.jsonl"],
};

fn member(split: SuperglueCopaSplit) -> &'static str {
	match split {
		SuperglueCopaSplit::Train => "train.jsonl",
		SuperglueCopaSplit::Validation => "val.jsonl",
		SuperglueCopaSplit::Test => "test.jsonl",
	}
}

/// SuperglueCopa JSONL record dataset over 4 text field(s).
pub struct SuperglueCopa {
	data: RecordData,
}

impl SuperglueCopa {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and JSONL decode errors.
	pub fn decode(config: &SuperglueCopaConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus
	/// `Cancelled` when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &SuperglueCopaConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => crate::core::pack::data_root()?,
		};
		let dir = crate::core::pack::ensure_pack_with_progress(&SUPERGLUE_COPA_PACK, &root, progress)?;
		let (decoded, _) = recordl::decode_member(
			"superglue_copa",
			&dir.join("COPA.zip"),
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
		config: &SuperglueCopaConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let data = RecordData::load_on(
			engine,
			&SUPERGLUE_COPA_PACK,
			std::path::Path::new("COPA.zip"),
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
			.ok_or_else(|| oa::Error::invalid_argument("superglue_copa field index out of range"))
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

impl Dataset for SuperglueCopa {
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
