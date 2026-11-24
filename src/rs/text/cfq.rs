//! CFQ compositional Freebase question-to-SPARQL pack (MCD1 split).
//!
//! `cfq.tar.gz` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `cfq/1.0.0`;
//! `cfq/dataset.json` carries the full example array and
//! `cfq/splits/mcd1.json` selects which indices form the maximum
//! compound-divergence split. Each example decodes into
//! `question`/`sparql` `[N, K_i]` F32 byte matrices; the pack carries no
//! labels. `get_item` returns the `[1, sum(K_i)]` end-to-end field row.

use oa::data::{Dataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::core::decode::Decoded;
use crate::core::extract;
use crate::core::pack::{self, FetchProgress};
use crate::text::recordl::{self, RecordData};

#[path = "cfq.gen.rs"]
mod generated;

pub(crate) use generated::CFQ_PACK;
pub use generated::{CfqConfig, CfqSplit};

const FIELDS: &[&str] = &["question", "sparql"];

#[derive(serde::Deserialize)]
struct CfqRow {
	question: String,
	sparql: String,
}

fn member(split: CfqSplit) -> &'static str {
	match split {
		CfqSplit::Train => "trainIdxs",
		CfqSplit::Validation => "devIdxs",
		CfqSplit::Test => "testIdxs",
	}
}

/// Decode `cfq/dataset.json` and the MCD1 index member selected by
/// `index_key` into `question`/`sparql` sample rows.
fn decode_member(
	pack: &str,
	tar_path: &std::path::Path,
	index_key: &str,
) -> Result<(Decoded, Vec<String>)> {
	let mut dataset: Option<Vec<CfqRow>> = None;
	let mut splits: Option<Vec<u8>> = None;
	extract::walk_tar_gz_reader(tar_path, |name, member| {
		if name == "cfq/dataset.json" {
			let buffered = std::io::BufReader::with_capacity(8 << 20, member);
			let mut parser = serde_json::Deserializer::from_reader(buffered);
			parser.disable_recursion_limit();
			dataset = Some(
				serde::Deserialize::deserialize(&mut parser).map_err(|error| {
					Error::data_loss(format!("{pack} dataset.json is not valid JSON: {error}"))
				})?,
			);
		} else if name == "cfq/splits/mcd1.json" {
			let mut data = Vec::new();
			member
				.read_to_end(&mut data)
				.map_err(|error| Error::io("read cfq/splits/mcd1.json", error))?;
			splits = Some(data);
		}
		Ok(())
	})?;
	let mut records =
		dataset.ok_or_else(|| Error::data_loss(format!("{pack} archive lacks cfq/dataset.json")))?;
	let splits =
		splits.ok_or_else(|| Error::data_loss(format!("{pack} archive lacks cfq/splits/mcd1.json")))?;
	let splits: serde_json::Value = serde_json::from_slice(&splits)
		.map_err(|error| Error::data_loss(format!("{pack} mcd1.json is not valid JSON: {error}")))?;
	let indices = splits
		.get(index_key)
		.and_then(|value| value.as_array())
		.ok_or_else(|| Error::data_loss(format!("{pack} mcd1.json lacks {index_key}")))?;
	let mut rows = Vec::with_capacity(indices.len());
	for index in indices {
		let index = index.as_u64().ok_or_else(|| {
			Error::data_loss(format!("{pack} mcd1.json {index_key} carries a non-index"))
		})? as usize;
		let row = records
			.get_mut(index)
			.ok_or_else(|| Error::data_loss(format!("{pack} dataset.json lacks example {index}")))?;
		rows.push(vec![
			std::mem::take(&mut row.question),
			std::mem::take(&mut row.sparql),
		]);
	}
	recordl::decode_rows(pack, FIELDS, &[], &[], false, &rows, &[], "json")
}

/// Cfq question/SPARQL pair dataset over 2 text field(s).
pub struct Cfq {
	data: RecordData,
}

impl Cfq {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and JSON decode errors.
	pub fn decode(config: &CfqConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus
	/// `Cancelled` when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &CfqConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&CFQ_PACK, &root, progress)?;
		let (decoded, _) = decode_member("cfq", &dir.join("cfq.tar.gz"), member(config.split))?;
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
		config: &CfqConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&CFQ_PACK, &root, progress)?;
		let (decoded, classes) = decode_member("cfq", &dir.join("cfq.tar.gz"), member(config.split))?;
		let data = RecordData::upload(engine, FIELDS, &decoded, classes, false)?;
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
			.ok_or_else(|| oa::Error::invalid_argument("cfq field index out of range"))
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

impl Dataset for Cfq {
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
