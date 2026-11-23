//! Standard MIDI File (SMF) parsing for note-sequence packs.
//!
//! [`parse`] reads format-0/1 files with pulse-per-quarter divisions, pairs
//! channel note-on/note-off events across tracks, and converts tick offsets
//! to seconds through the file's tempo map. Output is the modeling-friendly
//! `(pitch, step, duration)` event list the TensorFlow music-generation
//! tutorial derives from pretty_midi; control, sysex, lyric, and program
//! events are skipped, not decoded.

use oa::{Error, Result};

/// One sustained note, in seconds under the file's tempo map.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NoteEvent {
	/// MIDI key number `0..=127`.
	pub(crate) pitch: f32,
	/// Seconds between this note's onset and the previous note's onset;
	/// zero for the first event of a file.
	pub(crate) step: f32,
	/// Seconds the key stays down.
	pub(crate) duration: f32,
}

#[derive(Clone, Copy)]
struct PendingNote {
	start_tick: u64,
	open: bool,
}

/// Parse an SMF buffer into onset-ordered [`NoteEvent`]s.
///
/// # Errors
///
/// Returns `data_loss` when the header, track chunk, variable-length
/// quantity, or event stream is truncated or malformed, and
/// `invalid_argument` for format-2 files and SMPTE divisions.
pub(crate) fn parse(bytes: &[u8]) -> Result<Vec<NoteEvent>> {
	if bytes.len() < 14 || &bytes[..4] != b"MThd" {
		return Err(Error::data_loss("not a standard MIDI file"));
	}
	let header_len = be32(&bytes[4..8]) as usize;
	if header_len < 6
		|| 8usize
			.checked_add(header_len)
			.is_none_or(|end| end > bytes.len())
	{
		return Err(Error::data_loss("midi header is truncated"));
	}
	let format = be16(&bytes[8..10]);
	let tracks = be16(&bytes[10..12]) as usize;
	let division = be16(&bytes[12..14]);
	if format == 2 {
		return Err(Error::invalid_argument("midi format 2 is not supported"));
	}
	if division & 0x8000 != 0 {
		return Err(Error::invalid_argument(
			"midi SMPTE divisions are not supported",
		));
	}
	if division == 0 {
		return Err(Error::data_loss("midi division is zero"));
	}
	let ppq = f64::from(division);

	// (tick, us_per_quarter) tempo points and raw (start, end, pitch) spans.
	let mut tempos: Vec<(u64, u32)> = Vec::new();
	let mut spans: Vec<(u64, u64, u8)> = Vec::new();
	let mut offset = 8 + header_len;
	for _ in 0..tracks {
		if offset + 8 > bytes.len() || &bytes[offset..offset + 4] != b"MTrk" {
			return Err(Error::data_loss("midi track chunk is truncated"));
		}
		let track_len = be32(&bytes[offset + 4..offset + 8]) as usize;
		let end = offset
			.checked_add(8)
			.and_then(|start| start.checked_add(track_len))
			.ok_or_else(|| Error::data_loss("midi track length overflows"))?;
		if end > bytes.len() {
			return Err(Error::data_loss("midi track payload is truncated"));
		}
		read_track(&bytes[offset + 8..end], &mut tempos, &mut spans)?;
		offset = end;
	}
	if spans.is_empty() {
		return Err(Error::data_loss("midi file contains no note events"));
	}
	spans.sort_by_key(|&(start, _, _)| start);
	if tempos.is_empty() {
		tempos.push((0, 500_000));
	}
	tempos.sort_by_key(|&(tick, _)| tick);
	tempos.dedup_by_key(|entry| entry.0);
	if tempos.first().is_some_and(|&(tick, _)| tick > 0) {
		tempos.insert(0, (0, 500_000));
	}

	// Single sweep over the sorted span onsets/ends through the tempo map.
	let to_seconds = |tick: u64| -> f64 {
		let mut seconds = 0.0_f64;
		for (index, &(tempo_tick, us_per_quarter)) in tempos.iter().enumerate() {
			if tempo_tick >= tick {
				break;
			}
			let end = tempos
				.get(index + 1)
				.map_or(tick, |&(next, _)| next)
				.min(tick);
			seconds += f64::from(us_per_quarter) * (end - tempo_tick) as f64 / (ppq * 1.0e6);
		}
		seconds
	};

	let mut events = Vec::with_capacity(spans.len());
	let mut previous_start = 0.0_f64;
	for &(start_tick, end_tick, pitch) in &spans {
		let start = to_seconds(start_tick);
		let duration = (to_seconds(end_tick) - start).max(0.0);
		events.push(NoteEvent {
			pitch: f32::from(pitch),
			step: (start - previous_start) as f32,
			duration: duration as f32,
		});
		previous_start = start;
	}
	Ok(events)
}

fn read_track(
	track: &[u8],
	tempos: &mut Vec<(u64, u32)>,
	spans: &mut Vec<(u64, u64, u8)>,
) -> Result<()> {
	let mut pending = [PendingNote {
		start_tick: 0,
		open: false,
	}; 16 * 128];
	let mut pos = 0_usize;
	let mut tick = 0_u64;
	let mut status = 0_u8;
	let mut last_tick = 0_u64;
	while pos < track.len() {
		tick = tick
			.checked_add(read_varint(track, &mut pos)?)
			.ok_or_else(|| Error::data_loss("midi delta time overflows"))?;
		last_tick = tick;
		let byte = track[pos];
		if byte == 0xFF {
			// Meta event: type, varint length, payload.
			pos += 1;
			let kind = *track
				.get(pos)
				.ok_or_else(|| Error::data_loss("midi meta event is truncated"))?;
			pos += 1;
			let len = read_varint(track, &mut pos)? as usize;
			if pos + len > track.len() {
				return Err(Error::data_loss("midi meta payload is truncated"));
			}
			if kind == 0x51 && len == 3 {
				let us = (u32::from(track[pos]) << 16)
					| (u32::from(track[pos + 1]) << 8)
					| u32::from(track[pos + 2]);
				if us > 0 {
					tempos.push((tick, us));
				}
			}
			pos += len;
			if kind == 0x2F {
				break;
			}
		} else if byte == 0xF0 || byte == 0xF7 {
			pos += 1;
			let len = read_varint(track, &mut pos)? as usize;
			if pos + len > track.len() {
				return Err(Error::data_loss("midi sysex payload is truncated"));
			}
			pos += len;
		} else {
			let code = if byte & 0x80 != 0 {
				status = byte;
				pos += 1;
				byte >> 4
			} else {
				// Running status reuses the last channel event.
				if status == 0 {
					return Err(Error::data_loss(
						"midi running status without a status byte",
					));
				}
				status >> 4
			};
			let channel = usize::from(status & 0x0F);
			let data_len = if matches!(code, 0xC | 0xD) { 1 } else { 2 };
			if pos + data_len > track.len() {
				return Err(Error::data_loss("midi channel event is truncated"));
			}
			match code {
				0x8 | 0x9 => {
					let pitch = track[pos];
					let velocity = track[pos + 1];
					let slot = channel * 128 + usize::from(pitch);
					if code == 0x9 && velocity > 0 {
						pending[slot] = PendingNote {
							start_tick: tick,
							open: true,
						};
					} else if pending[slot].open {
						spans.push((
							pending[slot].start_tick,
							tick.max(pending[slot].start_tick),
							pitch,
						));
						pending[slot].open = false;
					}
				}
				_ => {}
			}
			pos += data_len;
		}
	}
	// Close notes still ringing at the track's last event.
	for (slot, note) in pending.iter_mut().enumerate() {
		if note.open && note.start_tick < last_tick {
			spans.push((note.start_tick, last_tick, (slot % 128) as u8));
		}
	}
	Ok(())
}

fn read_varint(bytes: &[u8], pos: &mut usize) -> Result<u64> {
	let mut value = 0_u64;
	for _ in 0..4 {
		let byte = *bytes
			.get(*pos)
			.ok_or_else(|| Error::data_loss("midi variable-length quantity is truncated"))?;
		*pos += 1;
		value = (value << 7) | u64::from(byte & 0x7F);
		if byte & 0x80 == 0 {
			return Ok(value);
		}
	}
	Err(Error::data_loss(
		"midi variable-length quantity exceeds 4 bytes",
	))
}

fn be16(bytes: &[u8]) -> u16 {
	u16::from_be_bytes([bytes[0], bytes[1]])
}

fn be32(bytes: &[u8]) -> u32 {
	u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[cfg(test)]
#[path = "../../../test/rs/audio/test_midi.rs"]
mod test_midi;
