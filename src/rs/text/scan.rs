//! SCAN compositional-generalization command-to-action pack.
//!
//! `scan-master.zip` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `scan/1.0.0`;
//! the simple-split task files decode each `IN: <command> OUT: <actions>`
//! line into `command`/`actions` `[N, K_i]` F32 byte matrices. The split
//! carries no labels. `get_item` returns the `[1, sum(K_i)]` end-to-end
//! field row.

use oa::data::{Dataset, Sample};
use oa::matrix::Matrix;
use oa::{Error, Result, Runtime};

use crate::core::decode::Decoded;
use crate::core::extract;
use crate::core::pack::{self, FetchProgress};
use crate::text::recordl::{self, RecordData};

#[path = "scan.gen.rs"]
mod generated;

pub(crate) use generated::SCAN_PACK;
pub use generated::{ScanConfig, ScanSplit};

const FIELDS: &[&str] = &["command", "actions"];

fn member(split: ScanSplit) -> &'static str {
	match split {
		ScanSplit::Train => "SCAN-master/simple_split/tasks_train_simple.txt",
		ScanSplit::Test => "SCAN-master/simple_split/tasks_test_simple.txt",
	}
}

/// Parse `IN: <command> OUT: <actions>` task lines into sample rows.
fn parse_rows(pack: &str, body: &[u8]) -> Result<Vec<Vec<String>>> {
	let text = std::str::from_utf8(body)
		.map_err(|_| Error::data_loss(format!("{pack} task file is not valid UTF-8")))?;
	let mut rows = Vec::new();
	for (line_no, line) in text.lines().enumerate() {
		if line.is_empty() {
			continue;
		}
		let pair = line
			.strip_prefix("IN: ")
			.and_then(|rest| rest.split_once(" OUT: "))
			.ok_or_else(|| {
				Error::data_loss(format!("{pack} task line {line_no} lacks IN:/OUT: fields"))
			})?;
		rows.push(vec![pair.0.to_owned(), pair.1.to_owned()]);
	}
	Ok(rows)
}

fn decode_member(
	pack: &str,
	zip_path: &std::path::Path,
	member: &str,
) -> Result<(Decoded, Vec<String>)> {
	let mut body: Option<Vec<u8>> = None;
	extract::walk_zip_file(
		zip_path,
		|name| name == member,
		|name, data| {
			if name == member {
				body = Some(data.to_vec());
			}
			Ok(())
		},
	)?;
	let body =
		body.ok_or_else(|| Error::data_loss(format!("{pack} archive lacks member {member}")))?;
	let rows = parse_rows(pack, &body)?;
	recordl::decode_rows(pack, FIELDS, &[], &[], false, &rows, &[], "task")
}

/// Scan command/action pair dataset over 2 text field(s).
pub struct Scan {
	data: RecordData,
}

impl Scan {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and task-file decode errors.
	pub fn decode(config: &ScanConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus
	/// `Cancelled` when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &ScanConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&SCAN_PACK, &root, progress)?;
		let (decoded, _) = decode_member("scan", &dir.join("scan-master.zip"), member(config.split))?;
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
		config: &ScanConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&SCAN_PACK, &root, progress)?;
		let (decoded, classes) =
			decode_member("scan", &dir.join("scan-master.zip"), member(config.split))?;
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
			.ok_or_else(|| oa::Error::invalid_argument("scan field index out of range"))
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

impl Dataset for Scan {
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
