//! Line-delimited JSON record reader for JSONL corpus members
//! (SuperGLUE tasks, ANLI, and similar packs).
//!
//! Each non-empty line parses to one [`serde_json::Value`]; pack
//! loaders map objects to text/label fields through their own
//! mappers so nested shapes (MultiRC passages, ReCoRD QA arrays,
//! WSC targets) stay pack-local.

use oa::{Error, Result};

/// Parse `bytes` as UTF-8 JSONL records. Empty lines are dropped.
///
/// # Errors
///
/// Returns `data_loss` when the payload is not valid UTF-8 or a line
/// is not valid JSON.
pub(crate) fn parse(pack: &str, bytes: &[u8]) -> Result<Vec<serde_json::Value>> {
	let text = std::str::from_utf8(bytes)
		.map_err(|_| Error::data_loss(format!("{pack} jsonl is not valid UTF-8")))?;
	text
		.lines()
		.filter(|line| !line.is_empty())
		.enumerate()
		.map(|(row, line)| {
			serde_json::from_str(line).map_err(|error| {
				Error::data_loss(format!("{pack} jsonl row {row} is not valid JSON: {error}"))
			})
		})
		.collect()
}

fn scalar_value(value: &serde_json::Value) -> Option<String> {
	match value {
		serde_json::Value::String(text) => Some(text.clone()),
		scalar @ (serde_json::Value::Number(_) | serde_json::Value::Bool(_)) => {
			Some(scalar.to_string())
		}
		_ => None,
	}
}

fn resolve<'a>(value: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
	let mut node = value;
	for segment in path.split('.') {
		node = node.get(segment)?;
	}
	Some(node)
}

/// Extract a required field at `path` (`.`-separated object keys) as
/// `String`; numbers and booleans render through their JSON spelling.
///
/// # Errors
///
/// Returns `data_loss` when `path` is absent or not a scalar.
pub(crate) fn scalar_at(
	pack: &str,
	value: &serde_json::Value,
	path: &str,
	row: usize,
) -> Result<String> {
	resolve(value, path).and_then(scalar_value).ok_or_else(|| {
		Error::data_loss(format!(
			"{pack} jsonl row {row} lacks scalar field '{path}'"
		))
	})
}

/// Extract an optional field at `path`; missing or non-scalar values
/// map to `""` (unlabeled test members).
pub(crate) fn scalar_at_opt(value: &serde_json::Value, path: &str) -> String {
	resolve(value, path)
		.and_then(scalar_value)
		.unwrap_or_default()
}
