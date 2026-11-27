//! Extreme Summarization (XSum) document/summary pairs.
//!
//! `xsum.tar.gz` and `xsum-split.json` are materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `xsum/1.0.0`.
//! Each `bbc-summary-data/<id>.summary` member carries `[SN]... [SN]`
//! sections; `FIRST-SENTENCE` is the one-line summary and `RESTBODY` the
//! article body. The EdinburghNLP manifest selects which ids form each
//! split (204,045 train / 11,332 validation / 11,334 test). Documents
//! flatten to a `[B]` byte blob with `[N]` `document_lengths` (offsets
//! exceed F32 precision, so callers derive spans by cumulative sum);
//! summaries pad to a `[N, K]` byte matrix. `get_item` returns the
//! `[1, len_i]` document row for the requested index.

use std::collections::HashSet;

use oa::data::{Dataset, Sample};
use oa::matrix::{self, Matrix};
use oa::{Error, Result, Runtime};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::extract;
use crate::core::pack::{self, FetchProgress};

#[path = "xsum.gen.rs"]
mod generated;

pub(crate) use generated::XSUM_PACK;
pub use generated::{XsumConfig, XsumSplit};

fn key(split: XsumSplit) -> &'static str {
	match split {
		XsumSplit::Train => "train",
		XsumSplit::Validation => "validation",
		XsumSplit::Test => "test",
	}
}

/// Parse one `.summary` member's `[SN]KEY[SN]` sections into
/// `(FIRST-SENTENCE, RESTBODY)`.
fn parse_summary(pack: &str, name: &str, body: &[u8]) -> Result<(String, String)> {
	let text = std::str::from_utf8(body)
		.map_err(|_| Error::data_loss(format!("{pack} member {name} is not valid UTF-8")))?;
	let mut first = String::new();
	let mut rest = String::new();
	let mut section = "";
	for line in text.lines() {
		if let Some(key) = line
			.strip_prefix("[SN]")
			.and_then(|l| l.strip_suffix("[SN]"))
		{
			section = key;
			continue;
		}
		match section {
			"FIRST-SENTENCE" => {
				if !first.is_empty() {
					first.push('\n');
				}
				first.push_str(line);
			}
			"RESTBODY" => {
				if !rest.is_empty() {
					rest.push('\n');
				}
				rest.push_str(line);
			}
			_ => {}
		}
	}
	if rest.is_empty() {
		return Err(Error::data_loss(format!(
			"{pack} member {name} lacks RESTBODY"
		)));
	}
	Ok((first, rest))
}

/// Decode the members selected by the split manifest into a flat
/// document blob plus a padded summary matrix.
fn decode_members(
	pack: &str,
	dir: &std::path::Path,
	split: XsumSplit,
	count: Option<usize>,
) -> Result<(Decoded, Vec<u64>)> {
	let manifest = std::fs::read_to_string(dir.join("xsum-split.json"))
		.map_err(|error| Error::io("read xsum split manifest", error))?;
	let manifest: serde_json::Value = serde_json::from_str(&manifest).map_err(|error| {
		Error::data_loss(format!("{pack} split manifest is not valid JSON: {error}"))
	})?;
	let wanted: HashSet<String> = manifest
		.get(key(split))
		.and_then(|value| value.as_array())
		.ok_or_else(|| Error::data_loss(format!("{pack} split manifest lacks {}", key(split))))?
		.iter()
		.filter_map(|value| value.as_str().map(str::to_owned))
		.collect();
	let mut blob: Vec<u8> = Vec::new();
	let mut doc_lengths: Vec<f32> = Vec::new();
	let mut offsets: Vec<u64> = Vec::new();
	let mut summaries: Vec<String> = Vec::new();
	extract::walk_tar_gz_reader(&dir.join("xsum.tar.gz"), |name, member| {
		let file = name.rsplit('/').next().unwrap_or(name);
		let id = file.strip_suffix(".summary");
		match id {
			Some(id) if wanted.contains(id) => {
				if count.is_some_and(|limit| offsets.len() >= limit) {
					return Err(Error::cancelled(format!("{pack} row cap reached")));
				}
				let mut data = Vec::new();
				member
					.read_to_end(&mut data)
					.map_err(|error| Error::io("read xsum member", error))?;
				let (first, rest) = parse_summary(pack, name, &data)?;
				offsets.push(blob.len() as u64);
				blob.extend_from_slice(rest.as_bytes());
				doc_lengths.push(rest.len() as f32);
				summaries.push(first);
			}
			_ => {}
		}
		Ok(())
	})?;
	if offsets.is_empty() {
		return Err(Error::data_loss(format!(
			"{pack} archive lacks {} members",
			key(split)
		)));
	}
	let width = summaries.iter().map(String::len).max().unwrap_or(0);
	let mut values = vec![0.0f32; summaries.len() * width];
	for (row, summary) in summaries.iter().enumerate() {
		for (k, &byte) in summary.as_bytes().iter().enumerate() {
			values[row * width + k] = f32::from(byte);
		}
	}
	let rows = doc_lengths.len();
	Ok((
		Decoded {
			matrices: vec![
				MatrixData {
					name: "document",
					shape: vec![blob.len()],
					values: blob.iter().map(|&byte| f32::from(byte)).collect(),
				},
				MatrixData {
					name: "document_lengths",
					shape: vec![rows],
					values: doc_lengths,
				},
				MatrixData {
					name: "summary",
					shape: vec![rows, width],
					values,
				},
			],
			text: None,
			seq_len: 0,
		},
		offsets,
	))
}

/// Xsum document/summary dataset: flat document blob, padded summary.
pub struct Xsum {
	document: Matrix,
	document_lengths: Matrix,
	summary: Matrix,
	offsets: Vec<u64>,
	rows: usize,
}

impl Xsum {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and member decode errors.
	pub fn decode(config: &XsumConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus
	/// `Cancelled` when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &XsumConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&XSUM_PACK, &root, progress)?;
		let (decoded, _) = decode_members("xsum", &dir, config.split, config.count)?;
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
		config: &XsumConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&XSUM_PACK, &root, progress)?;
		let (decoded, offsets) = decode_members("xsum", &dir, config.split, config.count)?;
		Ok(Self {
			rows: offsets.len(),
			document: decoded.upload("document", engine)?,
			document_lengths: decoded.upload("document_lengths", engine)?,
			summary: decoded.upload("summary", engine)?,
			offsets,
		})
	}

	/// `[B]` F32 flat document byte blob on the owning engine.
	#[must_use]
	pub const fn document(&self) -> &Matrix {
		&self.document
	}

	/// `[N]` F32 per-document byte lengths; cumulative sums give blob
	/// offsets.
	#[must_use]
	pub const fn document_lengths(&self) -> &Matrix {
		&self.document_lengths
	}

	/// `[N, K]` F32 padded summary byte matrix.
	#[must_use]
	pub const fn summary(&self) -> &Matrix {
		&self.summary
	}
}

impl Dataset for Xsum {
	fn len(&self) -> i64 {
		i64::try_from(self.rows).unwrap_or(i64::MAX)
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		let start = *self
			.offsets
			.get(index as usize)
			.ok_or_else(|| Error::invalid_argument("xsum index out of range"))?;
		let end = self
			.offsets
			.get(index as usize + 1)
			.copied()
			.unwrap_or(self.document.shape()[0] as u64);
		matrix::slice(
			&self.document,
			0,
			i64::try_from(start).unwrap_or(i64::MAX),
			i64::try_from(end).unwrap_or(i64::MAX),
		)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		Ok(Sample::new(self.get_item(index)?))
	}
}
