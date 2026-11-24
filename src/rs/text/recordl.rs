//! Shared JSONL record extraction for text classification and QA
//! tasks (SuperGLUE, ANLI).
//!
//! Each pack zip carries `<TASK>/{train,val,test}.jsonl` members; the
//! split selects which member decodes. A [`RecordSpec`] describes the
//! emitted contract: `fields` names the `[N, K_i]` F32 byte matrices
//! built from each text field (padded to that field's own maximum
//! width), a pack-local `map` flattens one JSON object into emitted
//! `(texts, label)` sample rows (one object may emit several
//! samples), and `label_names`/`classes` give class decoding and
//! display names. `unlabeled` member suffixes suppress the `labels`
//! matrix — SuperGLUE test members carry no labels.

use std::path::{Path, PathBuf};

use oa::data::Sample;
use oa::matrix::{self, Matrix};
use oa::{Engine, Error, Result};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::pack::{self, FetchProgress, Pack};
use crate::core::{extract, jsonl};

/// Flatten one parsed JSON object into `(texts, label)` sample
/// rows; `texts.len() == fields.len()`. `label` may be empty when
/// the member carries no labels.
pub(crate) type RowMap =
	fn(pack: &str, value: &serde_json::Value, row: usize) -> Result<Vec<(Vec<String>, String)>>;

/// Flattening contract for one JSONL text pack.
pub(crate) struct RecordSpec {
	/// Emitted text matrix names, in field order.
	pub fields: &'static [&'static str],
	/// Flatten one parsed JSON object into sample rows.
	pub map: RowMap,
	/// Member suffixes carrying no class labels (typically
	/// `test.jsonl`); matching members emit no `labels` matrix.
	pub unlabeled: &'static [&'static str],
	/// Ordered label strings for class-name decoding; empty means the
	/// label parses as a numeric class index or regression score.
	pub label_names: &'static [&'static str],
	/// Display names surfaced through `classes()`; index `i` names
	/// label `i`. Empty for regression tasks.
	pub classes: &'static [&'static str],
}

/// Extract `<task>/<member>` from `zip_path`, decode its JSONL
/// through `spec`, and return the resident matrix payload. Members
/// matching a `spec.unlabeled` suffix emit no `labels` matrix.
pub(crate) fn decode_member(
	pack: &str,
	zip_path: &Path,
	member: &str,
	spec: &RecordSpec,
) -> Result<(Decoded, Vec<String>)> {
	let has_label = !spec.unlabeled.iter().any(|s| member.ends_with(s));
	let suffix = format!("/{member}");
	let mut body: Option<Vec<u8>> = None;
	extract::walk_zip_file(
		zip_path,
		|name| name.ends_with(&suffix),
		|name, data| {
			if name.ends_with(&suffix) {
				body = Some(data.to_vec());
			}
			Ok(())
		},
	)?;
	let body =
		body.ok_or_else(|| Error::data_loss(format!("{pack} archive lacks member {member}")))?;
	let objects = jsonl::parse(pack, &body)?;

	let mut rows: Vec<Vec<String>> = Vec::new();
	let mut labels = Vec::new();
	for (row, object) in objects.iter().enumerate() {
		for (texts, label) in (spec.map)(pack, object, row)? {
			if texts.len() != spec.fields.len() {
				return Err(Error::data_loss(format!(
					"{pack} jsonl row {row} maps {} fields, expected {}",
					texts.len(),
					spec.fields.len()
				)));
			}
			rows.push(texts);
			labels.push(label);
		}
	}

	decode_rows(
		pack,
		spec.fields,
		spec.label_names,
		spec.classes,
		has_label,
		&rows,
		&labels,
		"jsonl",
	)
}

/// Build the resident matrices for `(texts, label)` sample rows from
/// any member format (JSONL, line pairs, per-member records): one
/// `[N, K_i]` F32 byte matrix per named text field, padded to that
/// field's own maximum width, plus a `[N]` F32 `labels` matrix when
/// `has_label`. `source` names the member format in error strings.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_rows(
	pack: &str,
	fields: &[&'static str],
	label_names: &[&str],
	classes: &[&str],
	has_label: bool,
	rows: &[Vec<String>],
	labels: &[String],
	source: &str,
) -> Result<(Decoded, Vec<String>)> {
	let mut widths = vec![0_usize; fields.len()];
	for texts in rows {
		if texts.len() != fields.len() {
			return Err(Error::data_loss(format!(
				"{pack} {source} row maps {} fields, expected {}",
				texts.len(),
				fields.len()
			)));
		}
		for (field, text) in texts.iter().enumerate() {
			widths[field] = widths[field].max(text.len());
		}
	}
	let mut matrices = Vec::with_capacity(fields.len() + 1);
	for (field, &name) in fields.iter().enumerate() {
		let mut values = vec![0.0f32; rows.len() * widths[field]];
		for (row, texts) in rows.iter().enumerate() {
			for (k, &byte) in texts[field].as_bytes().iter().enumerate() {
				values[row * widths[field] + k] = f32::from(byte);
			}
		}
		matrices.push(MatrixData {
			name,
			shape: vec![rows.len(), widths[field]],
			values,
		});
	}
	if has_label {
		if labels.len() != rows.len() {
			return Err(Error::data_loss(format!(
				"{pack} {source} carries {} labels for {} rows",
				labels.len(),
				rows.len()
			)));
		}
		let mut values = Vec::with_capacity(labels.len());
		for (row, cell) in labels.iter().enumerate() {
			let value = if label_names.is_empty() {
				cell.parse::<f32>().map_err(|_| {
					Error::data_loss(format!("{pack} {source} row {row} label is not numeric"))
				})?
			} else {
				label_names
					.iter()
					.position(|name| name == cell)
					.map(|index| index as f32)
					.ok_or_else(|| {
						Error::data_loss(format!(
							"{pack} {source} row {row} label '{cell}' is unknown"
						))
					})?
			};
			values.push(value);
		}
		matrices.push(MatrixData {
			name: "labels",
			shape: vec![values.len()],
			values,
		});
	}
	Ok((
		Decoded {
			matrices,
			text: None,
			seq_len: 0,
		},
		classes.iter().map(|name| (*name).to_owned()).collect(),
	))
}

/// Resident JSONL record payload: one `[N, K_i]` F32 byte matrix per
/// text field plus an optional `[N]` label matrix.
pub(crate) struct RecordData {
	/// Uploaded field matrices in `RecordSpec::fields` order.
	pub fields: Vec<Matrix>,
	pub labels: Option<Matrix>,
	pub classes: Vec<String>,
}

impl RecordData {
	/// Materialize `pack` under `cache_dir` (or the default data
	/// root), extract `member` from the task zip, and upload the
	/// resident matrices on `engine`.
	pub(crate) fn load_on(
		engine: &Engine,
		pack: &Pack,
		pack_dir: &Path,
		member: &str,
		spec: &'static RecordSpec,
		cache_dir: Option<PathBuf>,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let root = match &cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(pack, &root, progress)?;
		let has_label = !spec.unlabeled.iter().any(|s| member.ends_with(s));
		let (decoded, classes) = decode_member(pack.name, &dir.join(pack_dir), member, spec)?;
		Self::upload(engine, spec.fields, &decoded, classes, has_label)
	}

	/// Upload a decoded rows payload (any member source) on `engine`:
	/// one matrix per `fields` name plus `labels` when `has_label`.
	pub(crate) fn upload(
		engine: &Engine,
		fields: &[&str],
		decoded: &Decoded,
		classes: Vec<String>,
		has_label: bool,
	) -> Result<Self> {
		let fields = fields
			.iter()
			.map(|name| decoded.upload(name, engine))
			.collect::<Result<Vec<_>>>()?;
		let labels = if has_label {
			Some(decoded.upload("labels", engine)?)
		} else {
			None
		};
		Ok(Self {
			fields,
			labels,
			classes,
		})
	}
}

impl RecordData {
	pub(crate) fn rows(&self) -> usize {
		self
			.fields
			.first()
			.map(|field| field.shape()[0])
			.unwrap_or(0)
	}

	/// `[1, sum(K_i)]` byte row for sample `index`: every text field
	/// laid end to end in field order.
	pub(crate) fn item(&self, index: i64) -> Result<Matrix> {
		let rows = self
			.fields
			.iter()
			.map(|field| matrix::slice(field, 0, index, index + 1))
			.collect::<Result<Vec<_>>>()?;
		match rows.len() {
			0 => Err(Error::data_loss("record pack carries no text fields")),
			1 => Ok(rows.into_iter().next().unwrap_or_else(|| unreachable!())),
			_ => matrix::concat(&rows, 1),
		}
	}

	/// `(text row, label)` pair for sample `index`; the label is
	/// absent for unlabeled members.
	pub(crate) fn sample(&self, index: i64) -> Result<Sample> {
		let x = self.item(index)?;
		match &self.labels {
			Some(labels) => Ok(Sample::new_with_label(
				x,
				matrix::slice(labels, 0, index, index + 1)?,
			)),
			None => Ok(Sample::new(x)),
		}
	}
}
