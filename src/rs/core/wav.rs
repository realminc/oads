//! Canonical PCM WAV (RIFF/WAVE) parsing for clip packs.
//!
//! [`parse`] walks the RIFF chunk table and returns a borrowed [`WavView`]
//! describing the format chunk and the `data` payload. Loaders validate the
//! expected rate, channel count, and quantization against their pack's
//! contract; no resampling or format conversion happens here.

use oa::{Error, Result};

/// Borrowed view of one canonical WAV file's format fields and PCM payload.
pub(crate) struct WavView<'a> {
	/// Samples per second per channel.
	pub(crate) sample_rate: u32,
	/// Interleaved channel count.
	pub(crate) channels: u16,
	/// Quantization bits per sample.
	pub(crate) bits_per_sample: u16,
	/// Little-endian PCM payload, interleaved by channel.
	pub(crate) pcm: &'a [u8],
}

/// Parse a RIFF/WAVE buffer into a [`WavView`].
///
/// # Errors
///
/// Returns `data_loss` when the buffer is not WAVE, a chunk is truncated,
/// the `fmt ` or `data` chunk is missing, or the encoding is not PCM.
pub(crate) fn parse(bytes: &[u8]) -> Result<WavView<'_>> {
	if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
		return Err(Error::data_loss("not a RIFF/WAVE file"));
	}
	let mut offset = 12;
	let mut fmt = None;
	let mut pcm = None;
	while offset + 8 <= bytes.len() {
		let id = &bytes[offset..offset + 4];
		let size = le32(&bytes[offset + 4..offset + 8]) as usize;
		let start = offset + 8;
		let end = start
			.checked_add(size)
			.ok_or_else(|| Error::data_loss("wav chunk size overflows usize"))?;
		if end > bytes.len() {
			return Err(Error::data_loss("wav chunk extends past the file"));
		}
		match id {
			b"fmt " => fmt = Some(&bytes[start..end]),
			b"data" => {
				pcm = Some(&bytes[start..end]);
				break;
			}
			_ => {}
		}
		offset = end + (size & 1);
	}
	let fmt = fmt.ok_or_else(|| Error::data_loss("wav file has no fmt chunk"))?;
	let pcm = pcm.ok_or_else(|| Error::data_loss("wav file has no data chunk"))?;
	if fmt.len() < 16 || le16(&fmt[..2]) != 1 {
		return Err(Error::data_loss("wav file is not PCM encoded"));
	}
	Ok(WavView {
		channels: le16(&fmt[2..4]),
		sample_rate: le32(&fmt[4..8]),
		bits_per_sample: le16(&fmt[14..16]),
		pcm,
	})
}

fn le16(bytes: &[u8]) -> u16 {
	u16::from_le_bytes([bytes[0], bytes[1]])
}

fn le32(bytes: &[u8]) -> u32 {
	u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[cfg(test)]
#[path = "../../../test/rs/audio/test_wav.rs"]
mod test_wav;
