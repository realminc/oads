//! Shared IDX ubyte decoding for the MNIST family of packs.
//!
//! IDX stores a big-endian dimension header ahead of a raw unsigned-byte
//! payload: magic `0x803` for image grids and `0x801` for label vectors.
//! Decoding stays on the host; upload produces F32 matrices because
//! `oa::matrix::slice` is F32-only.

use oa::{Error, Result};

const IDX_IMAGES_MAGIC: u32 = 0x803;
const IDX_LABELS_MAGIC: u32 = 0x801;

/// Official split selection shared by every pack's `_native.load`
/// surface. Packs whose taxonomy has no validation arm reject the
/// variant with `invalid_argument`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IdxSplit {
	/// The training split.
	#[default]
	Train,
	/// The held-out evaluation split.
	Test,
	/// The intermediate validation split, where the pack carries one.
	Validation,
}

/// Decode an IDX-ubyte image member to `([N, 1, rows, cols], F32 pixels)`.
pub(crate) fn decode_images(
	pack: &str,
	bytes: &[u8],
	normalize: bool,
) -> Result<(Vec<usize>, Vec<f32>)> {
	if bytes.len() < 16 || be_u32(bytes, 0) != IDX_IMAGES_MAGIC {
		return Err(Error::data_loss(format!(
			"{pack} images member is not IDX-ubyte (0x803)"
		)));
	}
	let count = be_u32(bytes, 4) as usize;
	let rows = be_u32(bytes, 8) as usize;
	let cols = be_u32(bytes, 12) as usize;
	let expected = count
		.checked_mul(rows)
		.and_then(|pixels| pixels.checked_mul(cols))
		.ok_or_else(|| Error::data_loss(format!("{pack} IDX dimensions overflow usize")))?;
	let data = &bytes[16..];
	if data.len() != expected {
		return Err(Error::data_loss(format!(
			"{pack} images member holds {} pixels, expected {expected}",
			data.len()
		)));
	}
	let scale = if normalize { 1.0 / 255.0 } else { 1.0 };
	let values = data
		.iter()
		.map(|&v| f32::from(v) * scale)
		.collect::<Vec<_>>();
	Ok((vec![count, 1, rows, cols], values))
}

/// Decode an IDX-ubyte label member to `([N], F32 class indices)`.
pub(crate) fn decode_labels(pack: &str, bytes: &[u8]) -> Result<(Vec<usize>, Vec<f32>)> {
	if bytes.len() < 8 || be_u32(bytes, 0) != IDX_LABELS_MAGIC {
		return Err(Error::data_loss(format!(
			"{pack} labels member is not IDX-ubyte (0x801)"
		)));
	}
	let count = be_u32(bytes, 4) as usize;
	let data = &bytes[8..];
	if data.len() != count {
		return Err(Error::data_loss(format!(
			"{pack} labels member holds {} labels, expected {count}",
			data.len()
		)));
	}
	let values = data.iter().map(|&v| f32::from(v)).collect::<Vec<_>>();
	Ok((vec![count], values))
}

fn be_u32(bytes: &[u8], offset: usize) -> u32 {
	u32::from_be_bytes([
		bytes[offset],
		bytes[offset + 1],
		bytes[offset + 2],
		bytes[offset + 3],
	])
}
