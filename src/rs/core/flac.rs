//! FLAC STREAMINFO header parsing shared by lazy audio packs.
//!
//! Clip-length indexing must not decode every member; the leading
//! STREAMINFO block carries sample rate, channel count, and total
//! frame count in a fixed 42-byte window after the `fLaC` marker.

use oa::{Error, Result};

/// Sample rate, channel count, and total frame count from a FLAC
/// STREAMINFO block (always the first metadata block after `fLaC`).
///
/// # Errors
///
/// Returns `data_loss` when `data` is not a FLAC stream, lacks a
/// leading STREAMINFO block, or truncates the packed header.
pub(crate) fn streaminfo(pack: &str, data: &[u8]) -> Result<(u32, u32, u64)> {
	if data.len() < 42 || &data[..4] != b"fLaC" {
		return Err(Error::data_loss(format!(
			"{pack} clip is not a FLAC stream"
		)));
	}
	if data[4] & 0x7f != 0 {
		return Err(Error::data_loss(format!(
			"{pack} clip has no leading STREAMINFO"
		)));
	}
	let packed = u64::from_be_bytes(
		data[18..26]
			.try_into()
			.map_err(|_| Error::data_loss(format!("{pack} STREAMINFO is truncated")))?,
	);
	let rate = (packed >> 44) as u32;
	let channels = ((packed >> 41) & 0x7) as u32 + 1;
	let samples = packed & 0xF_FFFF_FFFF;
	Ok((rate, channels, samples))
}
