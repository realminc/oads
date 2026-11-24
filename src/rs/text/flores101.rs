//! FLORES-101 multilingual evaluation pairs (English source).
//!
//! `flores101_dataset.tar.gz` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `flores101/1.0.0`. For every non-English member the loader pairs
//! each line with the same-position line of `eng`: one sample per
//! `(english, target)` sentence pair, labelled by the 100 target
//! language codes. `dev` maps to Validation and `devtest` to Test;
//! FLORES carries no train split. `get_item` returns the
//! `[1, sum(K_i)]` end-to-end field row.

use oa::data::{Dataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::core::decode::Decoded;
use crate::core::extract;
use crate::core::pack::{self, FetchProgress};
use crate::text::recordl::{self, RecordData};

#[path = "flores101.gen.rs"]
mod generated;

pub(crate) use generated::FLORES101_PACK;
pub use generated::{Flores101Config, Flores101Split};

const FIELDS: &[&str] = &["english", "target"];

fn member(split: Flores101Split) -> &'static str {
	match split {
		Flores101Split::Validation => "dev",
		Flores101Split::Test => "devtest",
	}
}

/// Pair each non-English member's lines with `eng` into
/// `(english, target)` rows; the label is the target language index in
/// the sorted member list.
fn decode_member(
	pack: &str,
	tar_path: &std::path::Path,
	split: &str,
) -> Result<(Decoded, Vec<String>)> {
	let suffix = format!(".{split}");
	let mut english: Option<Vec<u8>> = None;
	let mut targets: Vec<(String, Vec<u8>)> = Vec::new();
	extract::walk_tar_gz(tar_path, |name, data| {
		let file = name.rsplit('/').next().unwrap_or(name);
		if !file.ends_with(&suffix) {
			return Ok(());
		}
		let lang = file.trim_end_matches(&suffix);
		if lang == "eng" {
			english = Some(data.to_vec());
		} else {
			targets.push((lang.to_owned(), data.to_vec()));
		}
		Ok(())
	})?;
	let english =
		english.ok_or_else(|| Error::data_loss(format!("{pack} archive lacks eng.{split}")))?;
	targets.sort_by(|a, b| a.0.cmp(&b.0));
	let classes: Vec<String> = targets.iter().map(|(lang, _)| lang.clone()).collect();
	let english = std::str::from_utf8(&english)
		.map_err(|_| Error::data_loss(format!("{pack} eng.{split} is not valid UTF-8")))?;
	let eng_lines: Vec<&str> = english.lines().collect();
	let mut rows = Vec::new();
	let mut labels = Vec::new();
	for (index, (lang, data)) in targets.iter().enumerate() {
		let text = std::str::from_utf8(data)
			.map_err(|_| Error::data_loss(format!("{pack} {lang}.{split} is not valid UTF-8")))?;
		let lines: Vec<&str> = text.lines().collect();
		if lines.len() != eng_lines.len() {
			return Err(Error::data_loss(format!(
				"{pack} {lang}.{split} carries {} lines, eng carries {}",
				lines.len(),
				eng_lines.len()
			)));
		}
		for (line, source) in lines.iter().zip(eng_lines.iter()) {
			rows.push(vec![(*source).to_owned(), (*line).to_owned()]);
			labels.push(index.to_string());
		}
	}
	recordl::decode_rows(
		pack,
		FIELDS,
		&[],
		&classes.iter().map(String::as_str).collect::<Vec<_>>(),
		true,
		&rows,
		&labels,
		"lines",
	)
	.map(|(decoded, _)| (decoded, classes))
}

/// Flores101 English/target pair dataset over 2 text field(s).
pub struct Flores101 {
	data: RecordData,
}

impl Flores101 {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and member decode errors.
	pub fn decode(config: &Flores101Config) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus
	/// `Cancelled` when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &Flores101Config,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&FLORES101_PACK, &root, progress)?;
		let (decoded, _) = decode_member(
			"flores101",
			&dir.join("flores101_dataset.tar.gz"),
			member(config.split),
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
		config: &Flores101Config,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&FLORES101_PACK, &root, progress)?;
		let (decoded, classes) = decode_member(
			"flores101",
			&dir.join("flores101_dataset.tar.gz"),
			member(config.split),
		)?;
		let data = RecordData::upload(engine, FIELDS, &decoded, classes, true)?;
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
			.ok_or_else(|| oa::Error::invalid_argument("flores101 field index out of range"))
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

impl Dataset for Flores101 {
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
