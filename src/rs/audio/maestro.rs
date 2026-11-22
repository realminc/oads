//! MAESTRO v3 MIDI performances as a windowed [`oa::data::Dataset`].
//!
//! The `maestro-v3.0.0-midi.zip` payload is materialized through
//! [`ensure_pack`](crate::pack::ensure_pack) under `maestro/3.0.0`, walked in
//! memory, and decoded on the host: every `.midi` member parses through
//! the `midi` parser into `(pitch, step, duration)` events in seconds under
//! that file's tempo map. Events concatenate into one `[E, 3]` F32 matrix;
//! the pack's CSV owns the official train/validation/test assignment, so a
//! [`MaestroSplit`] filters which performances contribute windows.
//!
//! The dataset item is a fixed `[seq_len, 3]` window taken by
//! [`oa::matrix::slice`] directly on the resident event matrix — window
//! offsets never leave a file boundary, so each item is a contiguous device
//! view with zero event duplication. [`Dataset::get_sample`] pairs the
//! window with the same-length window shifted one event, matching the
//! next-token framing of the TensorFlow music-generation tutorial.

use oa::data::{Dataset, Sample};
use oa::matrix::{self, Matrix};
use oa::{Engine, Error, Result};
use std::collections::HashMap;

use crate::decode::{Decoded, MatrixData};
use crate::extract;
use crate::midi;
use crate::pack::{self, FetchProgress};

#[path = "maestro.gen.rs"]
mod generated;

pub(crate) use generated::MAESTRO_PACK;
pub use generated::{DEFAULT_SEQ_LEN, MaestroConfig, MaestroSplit};

/// Device-resident MAESTRO window dataset.
///
/// `get_item`/`get_sample` index contiguous `[seq_len, 3]` event windows by
/// host-side offset table, so `oa::data::collate` produces `[B, seq_len, 3]`
/// batches without padding.
pub struct Maestro {
	notes: Matrix,
	offsets: Vec<i64>,
	seq_len: usize,
	split: MaestroSplit,
}

impl Maestro {
	/// Decode the pack on the host without uploading.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, zip member errors, and MIDI
	/// or CSV decode failures.
	pub fn decode(config: &MaestroConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled`
	/// when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &MaestroConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		if config.seq_len == 0 {
			return Err(Error::invalid_argument("maestro seq_len must be nonzero"));
		}
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&MAESTRO_PACK, &root, progress)?;
		let zip = std::fs::read(dir.join("maestro-v3.0.0-midi.zip"))
			.map_err(|error| Error::io("read maestro archive", error))?;

		// Pass one: the CSV assigns every performance to a split; collect the
		// member names and the split table before decoding any MIDI bytes.
		let mut csv = None;
		let mut members = Vec::new();
		extract::walk_zip(&zip, |name, data| {
			if name.ends_with("maestro-v3.0.0.csv") {
				csv = Some(data.to_vec());
			} else if is_midi_member(name) {
				members.push(name.to_string());
			}
			Ok(())
		})?;
		let csv = csv.ok_or_else(|| Error::data_loss("maestro pack is missing its CSV manifest"))?;
		let assignments = parse_split_csv(&csv)?;
		let wanted = members
			.iter()
			.filter(|name| {
				assignments
					.get(trim_member_dir(name))
					.is_some_and(|&assigned| assigned == config.split)
			})
			.cloned()
			.collect::<std::collections::HashSet<_>>();
		if wanted.is_empty() {
			return Err(Error::data_loss("maestro split selects no performances"));
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
				"maestro decode produced no complete windows",
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
		config: &MaestroConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Self::from_decoded(engine, &decoded, config)
	}

	fn from_decoded(engine: &Engine, decoded: &Decoded, config: &MaestroConfig) -> Result<Self> {
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
	pub fn split(&self) -> MaestroSplit {
		self.split
	}

	fn window(&self, index: i64) -> Result<i64> {
		self
			.offsets
			.get(index as usize)
			.copied()
			.ok_or_else(|| Error::out_of_range(format!("maestro window index {index}")))
	}
}

impl Dataset for Maestro {
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

fn trim_member_dir(name: &str) -> &str {
	name.rsplit('/').next().unwrap_or(name)
}

/// Read `split`/`midi_filename` columns out of the pack CSV; returns a
/// filename → [`MaestroSplit`] table. Fields may be RFC-4180 quoted.
fn parse_split_csv(csv: &[u8]) -> Result<HashMap<String, MaestroSplit>> {
	let text =
		String::from_utf8(csv.to_vec()).map_err(|_| Error::data_loss("maestro CSV is not UTF-8"))?;
	let mut lines = text.lines();
	let header = parse_csv_line(lines.next().unwrap_or_default());
	let name_col = header
		.iter()
		.position(|field| field == "midi_filename")
		.ok_or_else(|| Error::data_loss("maestro CSV has no midi_filename column"))?;
	let split_col = header
		.iter()
		.position(|field| field == "split")
		.ok_or_else(|| Error::data_loss("maestro CSV has no split column"))?;
	let mut table = HashMap::new();
	for line in lines {
		if line.is_empty() {
			continue;
		}
		let fields = parse_csv_line(line);
		let (Some(name), Some(split)) = (fields.get(name_col), fields.get(split_col)) else {
			return Err(Error::data_loss(
				"maestro CSV row has fewer fields than the header",
			));
		};
		let split = match split.as_str() {
			"train" => MaestroSplit::Train,
			"validation" => MaestroSplit::Validation,
			"test" => MaestroSplit::Test,
			other => {
				return Err(Error::data_loss(format!(
					"maestro CSV has unknown split '{other}'"
				)));
			}
		};
		table.insert(trim_member_dir(name).to_string(), split);
	}
	Ok(table)
}

fn parse_csv_line(line: &str) -> Vec<String> {
	let mut fields = Vec::new();
	let mut field = String::new();
	let mut quoted = false;
	let mut chars = line.chars().peekable();
	while let Some(ch) = chars.next() {
		match ch {
			'"' if quoted && chars.peek() == Some(&'"') => {
				field.push('"');
				chars.next();
			}
			'"' => quoted = !quoted,
			',' if !quoted => {
				fields.push(std::mem::take(&mut field));
			}
			_ => field.push(ch),
		}
	}
	fields.push(field);
	fields
}
