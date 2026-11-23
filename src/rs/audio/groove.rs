//! Groove MIDI Dataset drum performances as a windowed [`oa::data::Dataset`].
//!
//! `groove-v1.0.0-midionly.zip` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `groove/1.0.0`. Each
//! `groove/drummer*/session*/*.mid` member decodes through the crate SMF
//! parser into `(pitch, step, duration)` events — General MIDI drum note,
//! gap to the next onset, and sustain in seconds under the file's tempo map
//! — concatenated into one `[E, 3]` F32 matrix plus a window-offset table.
//! The pack `info.csv` owns the official train/validation/test assignment.
//! [`oa::data::Dataset::get_item`] returns a fixed `[seq_len, 3]` device
//! slice; windows never cross a performance boundary, so items are
//! contiguous views with no event duplication.

use oa::data::{Dataset, Sample};
use oa::matrix::{self, Matrix};
use oa::{Engine, Error, Result};
use std::collections::{HashMap, HashSet};

use crate::core::csv;
use crate::core::decode::{Decoded, MatrixData};
use crate::core::extract;
use crate::core::midi;
use crate::core::pack::{self, FetchProgress};

#[path = "groove.gen.rs"]
mod generated;

pub(crate) use generated::GROOVE_PACK;
pub use generated::{DEFAULT_SEQ_LEN, GrooveConfig, GrooveSplit};

/// Groove drum events as device-sliced next-event windows.
///
/// `notes()` is `[E, 3]` F32 — MIDI drum pitch, step to the next onset, and
/// duration in seconds — and each item is a `[seq_len, 3]` slice of it. The
/// label of [`oa::data::Dataset::get_sample`] is the same-length window
/// shifted one event ahead, the next-event prediction contract.
pub struct Groove {
	notes: Matrix,
	offsets: Vec<i64>,
	seq_len: usize,
	split: GrooveSplit,
}

impl Groove {
	/// Decode the selected split on the host without uploading.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, archive/CSV/MIDI decode
	/// errors, and `invalid_argument` when `seq_len` is zero or the split
	/// selects no performance with more than `seq_len` events.
	pub fn decode(config: &GrooveConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled`
	/// when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &GrooveConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		if config.seq_len == 0 {
			return Err(Error::invalid_argument("groove seq_len must be nonzero"));
		}
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&GROOVE_PACK, &root, progress)?;
		let zip = std::fs::read(dir.join("groove-v1.0.0-midionly.zip"))
			.map_err(|error| Error::io("read groove archive", error))?;

		// Pass one: info.csv assigns every performance to a split.
		let mut wanted = HashSet::new();
		let mut members = Vec::new();
		let assignments = split_table(&zip)?;
		extract::walk_zip(&zip, |name, _| {
			if let Some(rest) = name.strip_prefix("groove/")
				&& is_midi_member(rest)
			{
				members.push(rest.to_string());
			}
			Ok(())
		})?;
		for member in &members {
			if assignments.get(member) == Some(&config.split) {
				wanted.insert(format!("groove/{member}"));
			}
		}
		if wanted.is_empty() {
			return Err(Error::data_loss("groove split selects no performances"));
		}

		// Pass two: decode only the selected performances into one event
		// buffer plus a per-window offset table.
		let mut notes = Vec::new();
		let mut offsets = Vec::new();
		extract::walk_zip(&zip, |name, data| {
			if !wanted.contains(name) {
				return Ok(());
			}
			let events =
				midi::parse(data).map_err(|error| Error::data_loss(format!("decoding {name}: {error}")))?;
			let base = (notes.len() / 3) as f32;
			if events.len() > config.seq_len {
				for window in 0..events.len() - config.seq_len {
					offsets.push(base + window as f32);
				}
			}
			for event in &events {
				notes.extend_from_slice(&[event.pitch, event.step, event.duration]);
			}
			Ok(())
		})?;
		if offsets.is_empty() {
			return Err(Error::data_loss(
				"groove decode produced no complete windows",
			));
		}
		Ok(Decoded {
			matrices: vec![
				MatrixData {
					name: "notes",
					shape: vec![notes.len() / 3, 3],
					values: notes,
				},
				MatrixData {
					name: "windows",
					shape: vec![offsets.len()],
					values: offsets,
				},
			],
			text: None,
			seq_len: config.seq_len,
		})
	}

	/// [`load_on`](Self::load_on) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`load_on`](Self::load_on).
	pub fn load_with_progress_on(
		engine: &Engine,
		config: &GrooveConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		let notes = decoded.upload("notes", engine)?;
		let offsets = decoded
			.matrix("windows")?
			.values
			.iter()
			.map(|&offset| offset as i64)
			.collect();
		Ok(Self {
			notes,
			offsets,
			seq_len: decoded.seq_len,
			split: config.split,
		})
	}

	/// The resident `[E, 3]` `(pitch, step, duration)` event matrix.
	pub fn notes(&self) -> &Matrix {
		&self.notes
	}

	/// Events per window.
	pub fn seq_len(&self) -> usize {
		self.seq_len
	}

	/// The split these windows were drawn from.
	pub fn split(&self) -> GrooveSplit {
		self.split
	}

	fn window(&self, index: i64) -> Result<i64> {
		self
			.offsets
			.get(index as usize)
			.copied()
			.ok_or_else(|| Error::out_of_range(format!("groove window index {index}")))
	}
}

impl Dataset for Groove {
	fn len(&self) -> i64 {
		self.offsets.len() as i64
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		let offset = self.window(index)?;
		matrix::slice(&self.notes, 0, offset, offset + self.seq_len as i64)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		let offset = self.window(index)?;
		let x = matrix::slice(&self.notes, 0, offset, offset + self.seq_len as i64)?;
		let y = matrix::slice(&self.notes, 0, offset + 1, offset + 1 + self.seq_len as i64)?;
		Ok(Sample { x, y: Some(y) })
	}
}

fn is_midi_member(name: &str) -> bool {
	name.ends_with(".midi") || name.ends_with(".mid")
}

/// Read `split`/`midi_filename` columns out of the pack `info.csv`; returns a
/// member-path → [`GrooveSplit`] table.
fn split_table(zip: &[u8]) -> Result<HashMap<String, GrooveSplit>> {
	let mut csv_bytes = None;
	extract::walk_zip(zip, |name, data| {
		if name == "groove/info.csv" {
			csv_bytes = Some(data.to_vec());
		}
		Ok(())
	})?;
	let csv_bytes = csv_bytes.ok_or_else(|| Error::data_loss("groove pack is missing info.csv"))?;
	let records = csv::parse("groove", &csv_bytes)?;
	let mut rows = records.iter();
	let header = rows
		.next()
		.ok_or_else(|| Error::data_loss("groove info.csv is empty"))?;
	let name_col = header
		.iter()
		.position(|field| field == "midi_filename")
		.ok_or_else(|| Error::data_loss("groove info.csv has no midi_filename column"))?;
	let split_col = header
		.iter()
		.position(|field| field == "split")
		.ok_or_else(|| Error::data_loss("groove info.csv has no split column"))?;
	let mut table = HashMap::new();
	for record in rows {
		let (Some(name), Some(split)) = (record.get(name_col), record.get(split_col)) else {
			return Err(Error::data_loss(
				"groove info.csv row has fewer fields than the header",
			));
		};
		let split = match split.as_str() {
			"train" => GrooveSplit::Train,
			"validation" => GrooveSplit::Validation,
			"test" => GrooveSplit::Test,
			other => {
				return Err(Error::data_loss(format!(
					"groove info.csv has unknown split '{other}'"
				)));
			}
		};
		table.insert(name.clone(), split);
	}
	Ok(table)
}
