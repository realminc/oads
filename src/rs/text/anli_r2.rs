//! Adversarial NLI round 2 context-hypothesis classification task as a JSONL pack.
//!
//! `anli_v1.0.zip` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `anli_r2/1.0.0`
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

#[path = "anli_r2.gen.rs"]
mod generated;

pub(crate) use generated::ANLI_R2_PACK;
pub use generated::{AnliR2Config, AnliR2Split};

fn map(pack: &str, value: &serde_json::Value, row: usize) -> Result<Vec<(Vec<String>, String)>> {
	Ok(vec![(
		vec![
			jsonl::scalar_at(pack, value, "context", row)?,
			jsonl::scalar_at(pack, value, "hypothesis", row)?,
		],
		jsonl::scalar_at_opt(value, "label"),
	)])
}

const SPEC: RecordSpec = RecordSpec {
	fields: &["context", "hypothesis"],
	label_names: &["e", "n", "c"],
	classes: &["entailment", "neutral", "contradiction"],
	map,
	unlabeled: &["test.jsonl"],
};

fn member(split: AnliR2Split) -> &'static str {
	match split {
		AnliR2Split::Train => "R2/train.jsonl",
		AnliR2Split::Validation => "R2/dev.jsonl",
		AnliR2Split::Test => "R2/test.jsonl",
	}
}

/// AnliR2 JSONL record dataset over 2 text field(s).
pub struct AnliR2 {
	data: RecordData,
}

impl AnliR2 {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and JSONL decode errors.
	pub fn decode(config: &AnliR2Config) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus
	/// `Cancelled` when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &AnliR2Config,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => crate::core::pack::data_root()?,
		};
		let dir = crate::core::pack::ensure_pack_with_progress(&ANLI_R2_PACK, &root, progress)?;
		let (decoded, _) = recordl::decode_member(
			"anli_r2",
			&dir.join("anli_v1.0.zip"),
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
		config: &AnliR2Config,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let data = RecordData::load_on(
			engine,
			&ANLI_R2_PACK,
			std::path::Path::new("anli_v1.0.zip"),
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
			.ok_or_else(|| oa::Error::invalid_argument("anli_r2 field index out of range"))
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

impl Dataset for AnliR2 {
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
