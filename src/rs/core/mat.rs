//! MATLAB 5.0 `.mat` (v5) numeric-array reader shared by metadata packs.
//!
//! VGG-style bundles ship labels and split ids as `.mat` sidecars next to a
//! flat image tree (oxford_flowers102, cars196 devkits). The reader is
//! deliberately narrow: top-level `miMATRIX` and `miCOMPRESSED` elements are
//! walked, and each numeric matrix surfaces as a name → values map. Structs,
//! cells, sparse, and complex parts are rejected — split metadata does not
//! need them.

use std::collections::BTreeMap;
use std::io::Read;

use flate2::read::ZlibDecoder;
use oa::{Error, Result};

const MI_COMPRESSED: u32 = 15;
const MI_MATRIX: u32 = 14;
const MI_INT8: u32 = 1;
const MI_UINT8: u32 = 2;
const MI_INT16: u32 = 3;
const MI_UINT16: u32 = 4;
const MI_INT32: u32 = 5;
const MI_UINT32: u32 = 6;
const MI_SINGLE: u32 = 7;
const MI_DOUBLE: u32 = 9;
const MI_INT64: u32 = 12;
const MI_UINT64: u32 = 13;

/// Numeric matrix payload with its declared shape.
pub(crate) struct MatArray {
	/// Column-major dimensions as stored in the element.
	pub dims: Vec<usize>,
	/// Values promoted to f64, column-major order.
	pub values: Vec<f64>,
}

/// Read every named numeric matrix in a MAT 5.0 file.
///
/// # Errors
///
/// Returns `data_loss` when the v5 header, element framing, compression, or
/// matrix layout is malformed, and `invalid_argument`-class data errors when
/// a matrix is complex, non-numeric, or a struct/cell array.
pub(crate) fn read_arrays(pack: &str, bytes: &[u8]) -> Result<BTreeMap<String, MatArray>> {
	if bytes.len() < 128 || !bytes[..12].starts_with(b"MATLAB 5.0") {
		return Err(Error::data_loss(format!(
			"{pack} member is not a MAT 5.0 file"
		)));
	}
	// Bytes 126-127 carry the endian indicator ("IM" for little-endian).
	if &bytes[126..128] != b"IM" {
		return Err(Error::data_loss(format!(
			"{pack} mat file is not little-endian"
		)));
	}
	let mut out = BTreeMap::new();
	read_elements(pack, &bytes[128..], &mut out)?;
	Ok(out)
}

fn read_elements(pack: &str, mut bytes: &[u8], out: &mut BTreeMap<String, MatArray>) -> Result<()> {
	while !bytes.is_empty() {
		let ((ty, data), rest) = take_element(pack, &mut bytes)?;
		bytes = rest;
		match ty {
			MI_COMPRESSED => {
				let mut inner = Vec::new();
				ZlibDecoder::new(data)
					.read_to_end(&mut inner)
					.map_err(|e| Error::data_loss(format!("{pack} mat zlib element failed: {e}")))?;
				read_elements(pack, &inner, out)?;
			}
			MI_MATRIX => {
				let (name, array) = read_matrix(pack, data)?;
				out.insert(name, array);
			}
			_ => {
				return Err(Error::data_loss(format!(
					"{pack} mat element type {ty} is not supported"
				)));
			}
		}
	}
	Ok(())
}

/// One data element: type code, payload bytes, and the unread tail.
type Element<'a> = ((u32, &'a [u8]), &'a [u8]);

/// Split one element into (type, data, rest); `bytes` advances past it.
fn take_element<'a>(pack: &str, bytes: &mut &'a [u8]) -> Result<Element<'a>> {
	if bytes.len() < 8 {
		return Err(Error::data_loss(format!("{pack} mat element is truncated")));
	}
	let tag = le32(&bytes[..4]);
	if tag >> 16 != 0 {
		// Small-data-element: low u16 type, high u16 size, 4 inline bytes.
		let ty = tag & 0xffff;
		let size = (tag >> 16) as usize;
		let data = &bytes[4..8];
		let rest = &bytes[8..];
		if size > 4 {
			return Err(Error::data_loss(format!(
				"{pack} mat small element overruns its word"
			)));
		}
		return Ok(((ty, &data[..size]), rest));
	}
	let size = le32(&bytes[4..8]) as usize;
	let data_end = 8usize
		.checked_add(size)
		.ok_or_else(|| Error::data_loss(format!("{pack} mat element size overflows")))?;
	if bytes.len() < data_end {
		return Err(Error::data_loss(format!("{pack} mat element is truncated")));
	}
	// The spec pads element data to 8-byte boundaries, but scipy-era
	// writers skip padding on compressed payloads. Prefer the unpadded
	// offset when its next tag parses as an element.
	let padded_end = (data_end + 7) & !7;
	let end =
		if padded_end == data_end || data_end == bytes.len() || plausible_tag_at(bytes, data_end) {
			data_end
		} else {
			padded_end
		};
	if bytes.len() < end {
		return Err(Error::data_loss(format!("{pack} mat element is truncated")));
	}
	let data = &bytes[8..8 + size];
	let rest = &bytes[end..];
	Ok(((le32(&bytes[..4]), data), rest))
}

/// Whether the word at `offset` looks like a data-element tag: a
/// regular type (1–19) or a small-element pair within its 4-byte word.
fn plausible_tag_at(bytes: &[u8], offset: usize) -> bool {
	let Some(word) = bytes.get(offset..offset + 4) else {
		return false;
	};
	let tag = le32(word);
	let (ty, size) = (tag & 0xffff, tag >> 16);
	if size == 0 {
		return (1..=19).contains(&tag);
	}
	(1..=19).contains(&ty) && size <= 4
}

fn read_matrix(pack: &str, mut bytes: &[u8]) -> Result<(String, MatArray)> {
	// Array flags: miUINT32, two words; first byte of the first word is the
	// mx class. Complex is flagged in the second word (bit 3 of byte 0).
	let (flags_ty, flags) = sub_element(pack, &mut bytes)?;
	if flags_ty != MI_UINT32 || flags.len() < 8 {
		return Err(Error::data_loss(format!(
			"{pack} mat array flags are malformed"
		)));
	}
	let class = flags[0];
	// Bit 3 of the second flags word marks a complex array (imag part follows).
	if flags[4] & 0x08 != 0 {
		return Err(Error::data_loss(format!(
			"{pack} mat complex arrays are not supported"
		)));
	}
	if (1..=3).contains(&class) {
		return Err(Error::data_loss(format!(
			"{pack} mat struct/cell/function arrays are not supported"
		)));
	}
	let (dims_ty, dims_bytes) = sub_element(pack, &mut bytes)?;
	if dims_ty != MI_INT32 || dims_bytes.len() % 4 != 0 || dims_bytes.is_empty() {
		return Err(Error::data_loss(format!(
			"{pack} mat dims element is malformed"
		)));
	}
	let mut dims = Vec::with_capacity(dims_bytes.len() / 4);
	let mut count = 1usize;
	for chunk in dims_bytes.as_chunks::<4>().0 {
		let d = i32::from_le_bytes(*chunk);
		if d < 0 {
			return Err(Error::data_loss(format!("{pack} mat dim is negative")));
		}
		count = count
			.checked_mul(d as usize)
			.ok_or_else(|| Error::data_loss(format!("{pack} mat dims overflow")))?;
		dims.push(d as usize);
	}
	let (name_ty, name) = sub_element(pack, &mut bytes)?;
	if name_ty != MI_INT8 {
		return Err(Error::data_loss(format!(
			"{pack} mat name element is malformed"
		)));
	}
	let name = String::from_utf8_lossy(name).into_owned();
	if name.is_empty() {
		return Err(Error::data_loss(format!("{pack} mat matrix is unnamed")));
	}
	let (data_ty, data) = sub_element(pack, &mut bytes)?;
	let values = numeric_values(pack, data_ty, data, count)?;
	Ok((name, MatArray { dims, values }))
}

/// Pull the next sub-element inside a matrix body, honouring the
/// small-data-element framing.
fn sub_element<'a>(pack: &str, bytes: &mut &'a [u8]) -> Result<(u32, &'a [u8])> {
	let ((ty, data), rest) = take_element(pack, bytes)?;
	*bytes = rest;
	Ok((ty, data))
}

fn numeric_values(pack: &str, ty: u32, data: &[u8], count: usize) -> Result<Vec<f64>> {
	let need = match ty {
		MI_INT8 | MI_UINT8 => count,
		MI_INT16 | MI_UINT16 => count * 2,
		MI_INT32 | MI_UINT32 | MI_SINGLE => count * 4,
		MI_DOUBLE | MI_INT64 | MI_UINT64 => count * 8,
		_ => {
			return Err(Error::data_loss(format!(
				"{pack} mat data type {ty} is not numeric"
			)));
		}
	};
	if data.len() < need {
		return Err(Error::data_loss(format!(
			"{pack} mat data payload is truncated"
		)));
	}
	let mut out = Vec::with_capacity(count);
	match ty {
		MI_INT8 => out.extend(data[..count].iter().map(|&b| b as i8 as f64)),
		MI_UINT8 => out.extend(data[..count].iter().map(|&b| f64::from(b))),
		MI_INT16 => out.extend(
			data[..need]
				.as_chunks::<2>()
				.0
				.iter()
				.map(|b| i16::from_le_bytes(*b) as f64),
		),
		MI_UINT16 => out.extend(
			data[..need]
				.as_chunks::<2>()
				.0
				.iter()
				.map(|b| u16::from_le_bytes(*b) as f64),
		),
		MI_INT32 => out.extend(
			data[..need]
				.as_chunks::<4>()
				.0
				.iter()
				.map(|b| i32::from_le_bytes(*b) as f64),
		),
		MI_UINT32 => out.extend(
			data[..need]
				.as_chunks::<4>()
				.0
				.iter()
				.map(|b| u32::from_le_bytes(*b) as f64),
		),
		MI_SINGLE => out.extend(
			data[..need]
				.as_chunks::<4>()
				.0
				.iter()
				.map(|b| f64::from(f32::from_le_bytes(*b))),
		),
		MI_DOUBLE => out.extend(
			data[..need]
				.as_chunks::<8>()
				.0
				.iter()
				.map(|b| f64::from_le_bytes(*b)),
		),
		MI_INT64 => out.extend(
			data[..need]
				.as_chunks::<8>()
				.0
				.iter()
				.map(|b| i64::from_le_bytes(*b) as f64),
		),
		MI_UINT64 => out.extend(
			data[..need]
				.as_chunks::<8>()
				.0
				.iter()
				.map(|b| u64::from_le_bytes(*b) as f64),
		),
		_ => unreachable!(),
	}
	Ok(out)
}

fn le32(b: &[u8]) -> u32 {
	u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[cfg(test)]
#[path = "../../../test/rs/core/test_mat.rs"]
mod test;
