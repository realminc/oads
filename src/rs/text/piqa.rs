//! PhysicalIQA binary physical-commonsense task as a JSONL pack.
//!
//! Loose `*.jsonl`/`*.lst` files materialize through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `piqa/1.0.0`.
//! `train`/`valid` records carry `{goal, sol1, sol2}` and labels arrive
//! in the sibling `-labels.lst` file (one `0`/`1` per line); `tests`
//! ships unlabeled. Rows decode through `recordl` to one `[N, K_i]`
//! F32 byte matrix per field plus `[N]` labels; `get_item` returns the
//! `[1, sum(K_i)]` end-to-end field row.

use std::path::Path;

use oa::data::{Dataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::core::decode::Decoded;
use crate::core::jsonl;
use crate::core::pack::FetchProgress;
use crate::text::recordl::{self, RecordData};

#[path = "piqa.gen.rs"]
mod generated;

pub(crate) use generated::PIQA_PACK;
pub use generated::{PiqaConfig, PiqaSplit};

const FIELDS: &[&str] = &["goal", "sol1", "sol2"];

fn files(split: PiqaSplit) -> (&'static str, Option<&'static str>) {
	match split {
		PiqaSplit::Train => ("train.jsonl", Some("train-labels.lst")),
		PiqaSplit::Validation => ("valid.jsonl", Some("valid-labels.lst")),
		PiqaSplit::Test => ("tests.jsonl", None),
	}
}

fn decode_split(pack: &str, dir: &Path, split: PiqaSplit) -> Result<(Decoded, Vec<String>)> {
	let (jsonl_name, labels_name) = files(split);
	let body = std::fs::read(dir.join(jsonl_name))
		.map_err(|error| Error::io("read piqa jsonl member", error))?;
	let objects = jsonl::parse(pack, &body)?;
	let mut labels: Vec<String> = Vec::new();
	let has_label = labels_name.is_some();
	if let Some(name) = labels_name {
		let text = std::fs::read_to_string(dir.join(name))
			.map_err(|error| Error::io("read piqa labels member", error))?;
		labels = text.lines().map(str::trim).map(str::to_owned).collect();
		if labels.len() != objects.len() {
			return Err(Error::data_loss(format!(
				"{pack} label list {} holds {} rows for {} jsonl objects",
				name,
				labels.len(),
				objects.len()
			)));
		}
	}
	let mut rows = Vec::with_capacity(objects.len());
	for (row, object) in objects.iter().enumerate() {
		rows.push(vec![
			jsonl::scalar_at(pack, object, "goal", row)?,
			jsonl::scalar_at(pack, object, "sol1", row)?,
			jsonl::scalar_at(pack, object, "sol2", row)?,
		]);
		if !has_label {
			labels.push(String::new());
		}
	}
	recordl::decode_rows(
		pack,
		FIELDS,
		&["0", "1"],
		&["sol1", "sol2"],
		has_label,
		&rows,
		&labels,
		"jsonl",
	)
}

/// Piqa JSONL record dataset over 3 text field(s).
pub struct Piqa {
	data: RecordData,
	split: PiqaSplit,
}

impl Piqa {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and JSONL decode errors.
	pub fn decode(config: &PiqaConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus
	/// `Cancelled` when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &PiqaConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => crate::core::pack::data_root()?,
		};
		let dir = crate::core::pack::ensure_pack_with_progress(&PIQA_PACK, &root, progress)?;
		let (decoded, _) = decode_split("piqa", &dir, config.split)?;
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
		config: &PiqaConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => crate::core::pack::data_root()?,
		};
		let dir = crate::core::pack::ensure_pack_with_progress(&PIQA_PACK, &root, progress)?;
		let (decoded, classes) = decode_split("piqa", &dir, config.split)?;
		let has_label = matches!(config.split, PiqaSplit::Train | PiqaSplit::Validation);
		let data = RecordData::upload(engine, FIELDS, &decoded, classes, has_label)?;
		Ok(Self {
			data,
			split: config.split,
		})
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
			.ok_or_else(|| oa::Error::invalid_argument("piqa field index out of range"))
	}

	/// `[N]` F32 label matrix; `None` for the unlabeled test member.
	#[must_use]
	pub const fn labels(&self) -> Option<&Matrix> {
		self.data.labels.as_ref()
	}

	/// Class display names; label index `i` names `classes()[i]`.
	#[must_use]
	pub fn classes(&self) -> &[String] {
		&self.data.classes
	}

	/// Selected split.
	pub const fn split(&self) -> PiqaSplit {
		self.split
	}
}

impl Dataset for Piqa {
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
