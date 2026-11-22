//! Host-side SMF parser contract: note pairing, running status, tempo-map
//! timing, and malformed-input rejection.

use super::*;

fn varint(value: u32) -> Vec<u8> {
	let mut bytes = vec![(value & 0x7F) as u8];
	let mut value = value >> 7;
	while value > 0 {
		bytes.insert(0, ((value & 0x7F) as u8) | 0x80);
		value >>= 7;
	}
	bytes
}

fn meta(delta: u32, kind: u8, payload: &[u8]) -> Vec<u8> {
	let mut event = varint(delta);
	event.extend_from_slice(&[0xFF, kind]);
	event.extend_from_slice(&varint(payload.len() as u32));
	event.extend_from_slice(payload);
	event
}

fn event(delta: u32, bytes: &[u8]) -> Vec<u8> {
	let mut out = varint(delta);
	out.extend_from_slice(bytes);
	out
}

fn tempo(us_per_quarter: u32) -> Vec<u8> {
	meta(0, 0x51, &us_per_quarter.to_be_bytes()[1..])
}

fn midi_file(format: u16, division: u16, tracks: &[Vec<u8>]) -> Vec<u8> {
	let mut file = Vec::new();
	file.extend_from_slice(b"MThd");
	file.extend_from_slice(&6_u32.to_be_bytes());
	file.extend_from_slice(&format.to_be_bytes());
	file.extend_from_slice(&(tracks.len() as u16).to_be_bytes());
	file.extend_from_slice(&division.to_be_bytes());
	for track in tracks {
		file.extend_from_slice(b"MTrk");
		file.extend_from_slice(&(track.len() as u32).to_be_bytes());
		file.extend_from_slice(track);
	}
	file
}

#[test]
fn midi_pairs_note_events_in_seconds() {
	// 480 PPQ at the default 500000 us/quarter: 480 ticks is half a second.
	let track = [
		tempo(500_000),
		event(0, &[0x90, 60, 100]),
		event(480, &[0x80, 60, 0]),
		event(240, &[0x90, 64, 100]),
		event(480, &[0x80, 64, 0]),
		meta(0, 0x2F, &[]),
	]
	.concat();
	let events = parse(&midi_file(0, 480, &[track])).expect("parse");
	assert_eq!(events.len(), 2);
	assert_eq!(events[0].pitch, 60.0);
	assert!((events[0].duration - 0.5).abs() < 1e-6);
	assert!(events[0].step.abs() < 1e-6);
	assert_eq!(events[1].pitch, 64.0);
	assert!((events[1].step - 0.75).abs() < 1e-6);
	assert!((events[1].duration - 0.5).abs() < 1e-6);
}

#[test]
fn midi_running_status_and_velocity_zero_off() {
	// Second note-on omits its status byte; velocity-0 on acts as note-off.
	let track = [
		event(0, &[0x90, 60, 100]),
		event(480, &[0x90, 64, 100]),
		event(0, &[64, 0]),
		event(240, &[60, 0]),
		meta(0, 0x2F, &[]),
	]
	.concat();
	let events = parse(&midi_file(0, 480, &[track])).expect("parse");
	assert_eq!(events.len(), 2);
	assert!((events[0].step - 0.0).abs() < 1e-6);
	assert!((events[0].duration - 0.75).abs() < 1e-6);
	assert!((events[1].step - 0.5).abs() < 1e-6);
	assert!((events[1].duration - 0.0).abs() < 1e-6);
}

#[test]
fn midi_tempo_change_scales_later_events() {
	// Tempo doubles after tick 480: tick 960 is 0.5 s + 1.0 s later.
	let track0 = [
		tempo(500_000),
		meta(480, 0x51, &1_000_000_u32.to_be_bytes()[1..]),
	]
	.concat();
	let track1 = [
		event(0, &[0x90, 60, 100]),
		event(960, &[0x80, 60, 0]),
		meta(0, 0x2F, &[]),
	]
	.concat();
	let events = parse(&midi_file(1, 480, &[track0, track1])).expect("parse");
	assert_eq!(events.len(), 1);
	assert!((events[0].duration - 1.5).abs() < 1e-6);
}

#[test]
fn midi_rejects_malformed_and_unsupported_inputs() {
	assert!(parse(b"not midi").is_err());
	assert!(parse(&midi_file(2, 480, &[meta(0, 0x2F, &[])])).is_err());
	assert!(parse(&midi_file(0, 0x8000 | 25, &[meta(0, 0x2F, &[])])).is_err());
	assert!(parse(&midi_file(0, 480, &[meta(0, 0x2F, &[])])).is_err());
	let truncated = {
		let mut file = midi_file(0, 480, &[event(0, &[0x90, 60, 100])]);
		file.truncate(file.len() - 2);
		file
	};
	assert!(parse(&truncated).is_err());
	// Running status before any status byte is malformed.
	assert!(parse(&midi_file(0, 480, &[event(0, &[60, 100])])).is_err());
}
