//! Tab-separated field reader for TSV manifests (GLUE tasks and
//! similar corpora).
//!
//! Unlike the RFC 4180 [`csv`](crate::core::csv) reader, GLUE TSVs
//! carry no quoting: rows split on `\n`, fields on `\t`. A header row
//! is skipped on request.

use oa::{Error, Result};

/// Parse `bytes` as UTF-8 TSV records, skipping one header row when
/// `header` is set. Empty trailing rows are dropped.
///
/// # Errors
///
/// Returns `data_loss` when the payload is not valid UTF-8.
pub(crate) fn parse(pack: &str, bytes: &[u8], header: bool) -> Result<Vec<Vec<String>>> {
	let text = std::str::from_utf8(bytes)
		.map_err(|_| Error::data_loss(format!("{pack} tsv is not valid UTF-8")))?;
	let mut lines = text.lines();
	if header {
		lines.next();
	}
	Ok(
		lines
			.filter(|line| !line.is_empty())
			.map(|line| line.split('\t').map(str::to_owned).collect())
			.collect(),
	)
}
