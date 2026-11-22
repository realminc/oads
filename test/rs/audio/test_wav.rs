//! Host-side WAV parser contract: canonical RIFF/WAVE fmt+data extraction
//! and malformed-input rejection.

use super::*;

/// Canonical mono 16 kHz s16 WAV carrying `samples` little-endian i16s.
fn make_wav(samples: &[i16], rate: u32, channels: u16, bits: u16) -> Vec<u8> {
	let pcm_len = samples.len() * 2;
	let mut wav = Vec::new();
	wav.extend_from_slice(b"RIFF");
	wav.extend_from_slice(&((36 + pcm_len) as u32).to_le_bytes());
	wav.extend_from_slice(b"WAVE");
	wav.extend_from_slice(b"fmt ");
	wav.extend_from_slice(&16_u32.to_le_bytes());
	wav.extend_from_slice(&1_u16.to_le_bytes()); // PCM
	wav.extend_from_slice(&channels.to_le_bytes());
	wav.extend_from_slice(&rate.to_le_bytes());
	wav.extend_from_slice(&(rate * u32::from(channels) * u32::from(bits / 8)).to_le_bytes());
	wav.extend_from_slice(&(channels * (bits / 8)).to_le_bytes());
	wav.extend_from_slice(&bits.to_le_bytes());
	wav.extend_from_slice(b"data");
	wav.extend_from_slice(&(pcm_len as u32).to_le_bytes());
	for sample in samples {
		wav.extend_from_slice(&sample.to_le_bytes());
	}
	wav
}

#[test]
fn wav_parses_canonical_clip() {
	let wav = make_wav(&[0, 1_000, -1_000, 32_767, -32_768], 16_000, 1, 16);
	let view = parse(&wav).expect("parse");
	assert_eq!(view.sample_rate, 16_000);
	assert_eq!(view.channels, 1);
	assert_eq!(view.bits_per_sample, 16);
	assert_eq!(view.pcm.len(), 10);
	assert_eq!(i16::from_le_bytes([view.pcm[2], view.pcm[3]]), 1_000);
}

#[test]
fn wav_skips_unknown_chunks_before_data() {
	let mut wav = make_wav(&[7], 16_000, 1, 16);
	// Splice a LIST chunk between fmt and data.
	let data_at = wav
		.windows(4)
		.position(|w| w == b"data")
		.expect("data chunk");
	let mut spliced = wav[..data_at].to_vec();
	spliced.extend_from_slice(b"LIST");
	spliced.extend_from_slice(&8_u32.to_le_bytes());
	spliced.extend_from_slice(b"INFOJUNK");
	spliced.extend_from_slice(&wav[data_at..]);
	let view = parse(&spliced).expect("parse with LIST");
	assert_eq!(view.pcm.len(), 2);
	wav.clear();
}

#[test]
fn wav_rejects_non_wave_and_non_pcm() {
	assert!(parse(b"not a wav").is_err());
	let mut wav = make_wav(&[1], 16_000, 1, 16);
	wav[20] = 3; // format tag 3 = IEEE float, not PCM
	assert!(parse(&wav).is_err());
}
