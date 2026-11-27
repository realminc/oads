//! Word-in-Context word-sense disambiguation task as a SuperGLUE JSONL pack.
//!
//! `WiC.zip` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `superglue_wic/1.0.0`
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

#[path = "superglue_wic.gen.rs"]
mod generated;

pub(crate) use generated::SUPERGLUE_WIC_PACK;
pub use generated::{SuperglueWicConfig, SuperglueWicSplit};

fn map(pack: &str, value: &serde_json::Value, row: usize) -> Result<Vec<(Vec<String>, String)>> {
	Ok(vec![(
		vec![
			jsonl::scalar_at(pack, value, "word", row)?,
			jsonl::scalar_at(pack, value, "sentence1", row)?,
			jsonl::scalar_at(pack, value, "sentence2", row)?,
		],
		jsonl::scalar_at_opt(value, "label"),
	)])
}

const SPEC: RecordSpec = RecordSpec {
	fields: &["word", "sentence1", "sentence2"],
	label_names: &["false", "true"],
	classes: &["False", "True"],
	map,
	unlabeled: &["test.jsonl"],
};

fn member(split: SuperglueWicSplit) -> &'static str {
	match split {
		SuperglueWicSplit::Train => "train.jsonl",
		SuperglueWicSplit::Validation => "val.jsonl",
		SuperglueWicSplit::Test => "test.jsonl",
	}
}

/// SuperglueWic JSONL record dataset over 3 text field(s).
pub struct SuperglueWic {
	data: RecordData,
}

impl SuperglueWic {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and JSONL decode errors.
	pub fn decode(config: &SuperglueWicConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus
	/// `Cancelled` when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &SuperglueWicConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => crate::core::pack::data_root()?,
		};
		let dir = crate::core::pack::ensure_pack_with_progress(&SUPERGLUE_WIC_PACK, &root, progress)?;
		let (decoded, _) = recordl::decode_member(
			"superglue_wic",
			&dir.join("WiC.zip"),
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
		config: &SuperglueWicConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let data = RecordData::load_on(
			engine,
			&SUPERGLUE_WIC_PACK,
			std::path::Path::new("WiC.zip"),
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
			.ok_or_else(|| oa::Error::invalid_argument("superglue_wic field index out of range"))
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

impl Dataset for SuperglueWic {
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
