//! Shared GLUE-style TSV extraction for text-pair and single-text
//! classification tasks.
//!
//! Each GLUE task zip carries `<TASK>/{train,dev[,test]}.tsv`
//! members; the split selects which member decodes. A
//! [`GlueSpec`] describes the column layout: one or two text
//! columns padded into `[N, K]` F32 byte matrices, and one label
//! column read either as a class name (indexed through
//! `label_names`) or a free-floating regression score.

use std::path::{Path, PathBuf};

use oa::{Error, Result};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::{extract, tsv};

/// Column layout for one GLUE task TSV.
pub(crate) struct GlueSpec {
	/// TSV members carry a header row.
	pub header: bool,
	/// Zero-based text columns, one or two.
	pub text_cols: &'static [usize],
	/// Zero-based label column.
	pub label_col: usize,
	/// Label strings for tasks whose label column is a class name
	/// (MNLI/QNLI/RTE); empty when the column is numeric — class
	/// index for SST-2/CoLA/QQP/WNLI, regression score for STS-B.
	pub label_names: &'static [&'static str],
	/// Display names surfaced through `classes()`; index `i` names
	/// `classes[i]`. Empty for regression tasks (STS-B).
	pub classes: &'static [&'static str],
}

/// Extract `<task>/<member>` from `zip_path`, decode its TSV through
/// `spec`, and return the resident matrix payload. `names` is
/// `["text1"]` or `["text1", "text2"]` matching `spec.text_cols`.
pub(crate) fn decode_member(
	pack: &str,
	zip_path: &Path,
	member: &str,
	spec: &GlueSpec,
) -> Result<(Decoded, Vec<String>)> {
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
	let rows = tsv::parse(pack, &body, spec.header)?;
	Ok((
		decode_rows(pack, &rows, spec)?,
		spec.classes.iter().map(|name| (*name).to_owned()).collect(),
	))
}

/// Build the matrix payload from parsed TSV rows.
fn decode_rows(pack: &str, rows: &[Vec<String>], spec: &GlueSpec) -> Result<Decoded> {
	let need = spec
		.text_cols
		.iter()
		.chain(std::iter::once(&spec.label_col))
		.max()
		.copied()
		.unwrap_or(0)
		+ 1;
	let mut width = 0_usize;
	for (row, record) in rows.iter().enumerate() {
		if record.len() < need {
			return Err(Error::data_loss(format!(
				"{pack} tsv row {row} has {} fields, expected at least {need}",
				record.len()
			)));
		}
		for &col in spec.text_cols {
			width = width.max(record[col].len());
		}
	}
	let widths = vec![width; spec.text_cols.len()];
	let mut matrices = Vec::with_capacity(spec.text_cols.len() + 1);
	for (i, &col) in spec.text_cols.iter().enumerate() {
		let mut values = vec![0.0f32; rows.len() * widths[i]];
		for (row, record) in rows.iter().enumerate() {
			for (k, &byte) in record[col].as_bytes().iter().enumerate() {
				values[row * widths[i] + k] = f32::from(byte);
			}
		}
		matrices.push(MatrixData {
			name: if i == 0 { "text1" } else { "text2" },
			shape: vec![rows.len(), widths[i]],
			values,
		});
	}
	let mut labels = Vec::with_capacity(rows.len());
	for (row, record) in rows.iter().enumerate() {
		let cell = &record[spec.label_col];
		let value = if spec.label_names.is_empty() {
			cell
				.parse::<f32>()
				.map_err(|_| Error::data_loss(format!("{pack} tsv row {row} label is not numeric")))?
		} else {
			spec
				.label_names
				.iter()
				.position(|name| name == cell)
				.map(|index| index as f32)
				.ok_or_else(|| {
					Error::data_loss(format!("{pack} tsv row {row} label '{cell}' is unknown"))
				})?
		};
		labels.push(value);
	}
	matrices.push(MatrixData {
		name: "labels",
		shape: vec![labels.len()],
		values: labels,
	});
	Ok(Decoded {
		matrices,
		text: None,
		seq_len: 0,
	})
}

use oa::Runtime;
use oa::data::Sample;
use oa::matrix::{self, Matrix};

use crate::core::pack::{self, FetchProgress, Pack};

/// Resident GLUE TSV payload: one or two `[N, K]` F32 text byte
/// matrices plus the `[N]` label/score matrix.
pub(crate) struct GlueData {
	pub text1: Matrix,
	pub text2: Option<Matrix>,
	pub labels: Matrix,
	pub classes: Vec<String>,
}

impl GlueData {
	/// Materialize `pack` under `cache_dir` (or the default data
	/// root), extract `member` from the task zip, and upload the
	/// resident matrices on `engine`.
	pub(crate) fn load_on(
		engine: &Runtime,
		pack: &Pack,
		pack_dir: &Path,
		member: &str,
		spec: &GlueSpec,
		cache_dir: Option<PathBuf>,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let root = match &cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(pack, &root, progress)?;
		let (decoded, classes) = decode_member(pack.name, &dir.join(pack_dir), member, spec)?;
		Ok(Self {
			text1: decoded.upload("text1", engine)?,
			text2: decoded
				.matrix("text2")
				.ok()
				.map(|_| decoded.upload("text2", engine))
				.transpose()?,
			labels: decoded.upload("labels", engine)?,
			classes,
		})
	}
}

impl GlueData {
	pub(crate) fn rows(&self) -> usize {
		self.text1.shape()[0]
	}

	/// `[F, K]` stacked text rows for sample `index` (F = 1 or 2).
	pub(crate) fn item(&self, index: i64) -> Result<Matrix> {
		let first = matrix::slice(&self.text1, 0, index, index + 1)?;
		match &self.text2 {
			Some(second) => {
				let rest = matrix::slice(second, 0, index, index + 1)?;
				matrix::concat(&[first, rest], 0)
			}
			None => Ok(first),
		}
	}

	/// `(text rows, label)` pair for sample `index`.
	pub(crate) fn sample(&self, index: i64) -> Result<Sample> {
		let x = self.item(index)?;
		let y = matrix::slice(&self.labels, 0, index, index + 1)?;
		Ok(Sample::new_with_label(x, y))
	}
}
