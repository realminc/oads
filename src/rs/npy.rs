//! NumPy `.npy` v1/v2 header decoding shared by array-format packs.
//!
//! Parsing stays header-only: the shape, dtype descriptor, and payload
//! borrow come out of one pass. Only C-order (`fortran_order = False`)
//! little-endian u8 payloads are accepted; loaders convert to F32 on upload
//! because `oa::matrix::slice` is F32-only.

use oa::{Error, Result};

/// Borrowed `.npy` contents.
pub(crate) struct NpyFile<'a> {
	/// Logical array shape from the header tuple.
	pub shape: Vec<usize>,
	/// Raw C-order payload bytes (u8 elements).
	pub data: &'a [u8],
}

/// Parse the `.npy` header and borrow the u8 payload.
///
/// # Errors
///
/// Returns `data_loss` when the magic, version, header dict, shape tuple, or
/// payload length is malformed, or when `descr`/`fortran_order` describe a
/// layout this reader does not accept.
pub(crate) fn read<'a>(pack: &str, bytes: &'a [u8]) -> Result<NpyFile<'a>> {
	if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
		return Err(Error::data_loss(format!(
			"{pack} member is not an npy array"
		)));
	}
	let (major, header_offset) = match bytes[6] {
		1 => (1, 8),
		2 | 3 => (bytes[6] as usize, 8),
		other => {
			return Err(Error::data_loss(format!(
				"{pack} npy version {other} is not supported"
			)));
		}
	};
	let header_len = if major == 1 {
		u16::from_le_bytes([bytes[header_offset], bytes[header_offset + 1]]) as usize
	} else {
		u32::from_le_bytes([
			bytes[header_offset],
			bytes[header_offset + 1],
			bytes[header_offset + 2],
			bytes[header_offset + 3],
		]) as usize
	};
	let dict_start = header_offset + if major == 1 { 2 } else { 4 };
	let dict_end = dict_start + header_len;
	if dict_end > bytes.len() {
		return Err(Error::data_loss(format!("{pack} npy header is truncated")));
	}
	let header = std::str::from_utf8(&bytes[dict_start..dict_end])
		.map_err(|_| Error::data_loss(format!("{pack} npy header is not UTF-8")))?;
	if !header.contains("'fortran_order': False") {
		return Err(Error::data_loss(format!(
			"{pack} npy member is not C-order"
		)));
	}
	if !header.contains("'descr': '|u1'") && !header.contains("'descr': '<u1'") {
		return Err(Error::data_loss(format!(
			"{pack} npy member is not u8 (|u1/<u1)"
		)));
	}
	let shape = parse_shape(pack, header)?;
	let expected = shape
		.iter()
		.try_fold(1_usize, |acc, &dim| acc.checked_mul(dim))
		.ok_or_else(|| Error::data_loss(format!("{pack} npy shape overflows usize")))?;
	let data = &bytes[dict_end..];
	if data.len() != expected {
		return Err(Error::data_loss(format!(
			"{pack} npy member holds {} bytes, expected {expected}",
			data.len()
		)));
	}
	Ok(NpyFile { shape, data })
}

fn parse_shape(pack: &str, header: &str) -> Result<Vec<usize>> {
	let start = header
		.find("'shape':")
		.and_then(|at| header[at..].find('(').map(|open| at + open + 1))
		.ok_or_else(|| Error::data_loss(format!("{pack} npy header has no shape")))?;
	let end = header[start..]
		.find(')')
		.map(|close| start + close)
		.ok_or_else(|| Error::data_loss(format!("{pack} npy shape is truncated")))?;
	header[start..end]
		.split(',')
		.filter(|part| !part.trim().is_empty())
		.map(|part| {
			part
				.trim()
				.parse::<usize>()
				.map_err(|_| Error::data_loss(format!("{pack} npy shape dim is not an integer")))
		})
		.collect()
}
