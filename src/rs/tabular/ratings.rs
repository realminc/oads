//! Shared MovieLens rating-record decode for the `u.data`/`u1.*` TSV and
//! `ratings.dat` `::`-separated layouts.

use oa::{Error, Result};

/// One rating record: `user_id`, `item_id`, `rating`, `timestamp`.
pub(crate) struct Rating {
	pub(crate) user_id: f32,
	pub(crate) item_id: f32,
	pub(crate) rating: f32,
	pub(crate) timestamp: f32,
}

/// Parse UTF-8 rating lines split on `delim` into [`Rating`] records.
///
/// # Errors
///
/// Returns `data_loss` for invalid UTF-8, a record without exactly four
/// fields, or a non-numeric field.
pub(crate) fn parse(pack: &str, bytes: &[u8], delim: &str) -> Result<Vec<Rating>> {
	let text = std::str::from_utf8(bytes)
		.map_err(|_| Error::data_loss(format!("{pack} ratings are not valid UTF-8")))?;
	let mut ratings = Vec::new();
	for (line_no, line) in text.lines().enumerate() {
		if line.is_empty() {
			continue;
		}
		let fields: Vec<&str> = line.split(delim).collect();
		if fields.len() != 4 {
			return Err(Error::data_loss(format!(
				"{pack} ratings line {} has {} fields, expected 4",
				line_no + 1,
				fields.len()
			)));
		}
		let mut numeric = [0.0f32; 4];
		for (index, field) in fields.iter().enumerate() {
			numeric[index] = field.parse::<f32>().map_err(|_| {
				Error::data_loss(format!(
					"{pack} ratings line {} field '{field}' is not numeric",
					line_no + 1
				))
			})?;
		}
		ratings.push(Rating {
			user_id: numeric[0],
			item_id: numeric[1],
			rating: numeric[2],
			timestamp: numeric[3],
		});
	}
	Ok(ratings)
}
