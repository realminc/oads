//! RFC 4180 CSV field parser for dataset manifests and corpora.
//!
//! [`parse`] yields one `Vec<String>` per record. Quoted fields may contain
//! commas, CR/LF pairs, and doubled `""` escapes. Unquoted fields end at the
//! first comma or newline. The parser is streaming-line-based: quoted
//! newlines are consumed inside the field, so record boundaries are
//! newlines only outside quotes.

use oa::{Error, Result};

/// Parse `bytes` as UTF-8 CSV records.
///
/// # Errors
///
/// Returns `data_loss` for invalid UTF-8 or an unterminated quoted field.
pub(crate) fn parse(pack: &str, bytes: &[u8]) -> Result<Vec<Vec<String>>> {
	let text = std::str::from_utf8(bytes)
		.map_err(|_| Error::data_loss(format!("{pack} csv is not valid UTF-8")))?;
	let mut records = Vec::new();
	let mut fields = Vec::new();
	let mut field = String::new();
	let mut chars = text.chars().peekable();
	let mut quoted = false;
	let mut field_ended = false;
	while let Some(c) = chars.next() {
		if quoted {
			match c {
				'"' if chars.peek() == Some(&'"') => {
					chars.next();
					field.push('"');
				}
				'"' => {
					quoted = false;
					field_ended = true;
				}
				_ => field.push(c),
			}
		} else {
			match c {
				'"' if field.is_empty() && !field_ended => quoted = true,
				',' => {
					fields.push(std::mem::take(&mut field));
					field_ended = false;
				}
				'\n' => {
					fields.push(std::mem::take(&mut field));
					records.push(std::mem::take(&mut fields));
					field_ended = false;
				}
				'\r' => {}
				_ => {
					field.push(c);
					field_ended = false;
				}
			}
		}
	}
	if quoted {
		return Err(Error::data_loss(format!(
			"{pack} csv ends inside a quoted field"
		)));
	}
	if !field.is_empty() || !fields.is_empty() || field_ended {
		fields.push(field);
		records.push(fields);
	}
	Ok(records)
}
