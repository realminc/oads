//! NSynth test set as a lazy [`oa::data::Dataset`].
//!
//! `nsynth-test.jsonwav.tar.gz` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `nsynth/1.0.0`. First load extracts the 4,096 `audio/*.wav` notes
//! to `extracted/wav/` and the `examples.json` metadata table to
//! `extracted/examples.json`. Only the `[N]` F32 family labels and the
//! `[N]` pitch, velocity, instrument, and sample-count matrices are
//! resident; [`oa::data::Dataset::get_item`] lazily decodes one
//! `[1, T_i]` F32 mono waveform through [`oa::audio`].
//!
//! The supervised pair is a lazily decoded note -> the canonical
//! instrument-family index into [`FAMILIES`]; pitch, velocity, and
//! per-instrument ids ride along for the instrument-classification
//! and pitch-regression tasks.

use std::path::PathBuf;

use oa::data::{Dataset, Sample};
use oa::matrix::{self, Matrix};
use oa::{Engine, Error, Result};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::pack::{self, FetchProgress};
use crate::core::{extract, wav};

#[path = "nsynth.gen.rs"]
mod generated;

pub(crate) use generated::NSYNTH_PACK;
pub use generated::NsynthConfig;

const PACK_DIR: &str = "nsynth/1.0.0";
const SAMPLE_RATE: u32 = 16_000;

/// Instrument-family names in canonical index order; label index `i`
/// names `FAMILIES[i]`.
pub const FAMILIES: [&str; 11] = [
	"bass",
	"brass",
	"flute",
	"guitar",
	"keyboard",
	"mallet",
	"organ",
	"reed",
	"string",
	"synth_lead",
	"vocal",
];

/// One `examples.json` note record.
struct Note {
	note_str: String,
	pitch: f32,
	velocity: f32,
	instrument: f32,
	family: f32,
	samples: f32,
}

/// NSynth test set: 4,096 four-second notes across 11 instrument
/// families.
///
/// `get_item` lazily decodes a `[1, T_i]` F32 mono waveform at
/// 16,000 Hz; `labels()` is the resident `[N]` F32
/// instrument-family index into [`FAMILIES`], and `pitches()`,
/// `velocities()`, `instruments()`, and `lengths()` carry the
/// remaining per-note metadata.
pub struct Nsynth {
	labels: Matrix,
	pitches: Matrix,
	velocities: Matrix,
	instruments: Matrix,
	lengths: Matrix,
	notes: Vec<String>,
	wav_dir: PathBuf,
	engine: Engine,
}

impl Nsynth {
	/// Fetch, verify, extract, and decode the pack on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, tar decode errors, JSON
	/// metadata errors, WAV header errors, and I/O errors while
	/// materializing the extracted clip cache.
	pub fn decode(config: &NsynthConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled`
	/// when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &NsynthConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&NSYNTH_PACK, &root, progress)?;
		let extracted = dir.join("extracted");
		if !extracted.join("index.csv").exists() {
			extract_archive(&dir, &extracted)?;
		}
		let notes = read_index(&extracted)?;
		Ok(decoded(notes))
	}

	/// [`load_on`](Self::load_on) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`load_on`](Self::load_on), plus `Cancelled`
	/// when the callback aborts a fetch.
	pub fn load_with_progress_on(
		engine: &Engine,
		config: &NsynthConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let decoded = Self::decode_with_progress(config, progress)?;
		let extracted = root.join(PACK_DIR).join("extracted");
		let notes = read_index(&extracted)?;
		Ok(Self {
			labels: decoded.upload("labels", engine)?,
			pitches: decoded.upload("pitches", engine)?,
			velocities: decoded.upload("velocities", engine)?,
			instruments: decoded.upload("instruments", engine)?,
			lengths: decoded.upload("lengths", engine)?,
			notes: notes.iter().map(|note| note.note_str.clone()).collect(),
			wav_dir: extracted.join("wav"),
			engine: engine.clone(),
		})
	}

	/// Instrument-family names in canonical index order; label index
	/// `i` names `families()[i]`.
	#[must_use]
	pub fn families(&self) -> &[&'static str] {
		&FAMILIES
	}

	/// Full `[N]` F32 instrument-family index matrix into
	/// [`families`](Self::families).
	#[must_use]
	pub const fn labels(&self) -> &Matrix {
		&self.labels
	}

	/// Full `[N]` F32 MIDI pitch matrix (0–127).
	#[must_use]
	pub const fn pitches(&self) -> &Matrix {
		&self.pitches
	}

	/// Full `[N]` F32 MIDI velocity matrix (0–127 in quantized steps
	/// of 25).
	#[must_use]
	pub const fn velocities(&self) -> &Matrix {
		&self.velocities
	}

	/// Full `[N]` F32 upstream instrument id matrix.
	#[must_use]
	pub const fn instruments(&self) -> &Matrix {
		&self.instruments
	}

	/// Full `[N]` F32 clip sample counts at 16,000 Hz.
	#[must_use]
	pub const fn lengths(&self) -> &Matrix {
		&self.lengths
	}

	/// Upstream `<instrument>-<pitch>-<velocity>` id of row `index`.
	///
	/// # Errors
	///
	/// Returns `out_of_range` when `index` is outside the dataset.
	pub fn note_id(&self, index: i64) -> Result<&str> {
		if index < 0 || index >= self.len() {
			return Err(Error::out_of_range("nsynth index is outside the dataset"));
		}
		Ok(&self.notes[index as usize])
	}

	/// Lazily decode note `index` to a `[1, T_i]` F32 mono waveform on
	/// the owning engine.
	///
	/// # Errors
	///
	/// Returns `out_of_range` for an out-of-range index and the
	/// [`oa::audio::decode_file_on`] errors when the extracted clip is
	/// not 16,000 Hz mono PCM.
	pub fn waveform(&self, index: i64) -> Result<Matrix> {
		let file = self.note_id(index)?;
		let audio = oa::audio::decode_file_on(&self.engine, self.wav_dir.join(format!("{file}.wav")))?;
		if audio.sample_rate() != SAMPLE_RATE || audio.as_matrix().shape()[0] != 1 {
			return Err(Error::data_loss(format!(
				"nsynth note {file} is not {SAMPLE_RATE} Hz mono WAV"
			)));
		}
		Ok(audio.as_matrix().clone())
	}
}

impl Dataset for Nsynth {
	fn len(&self) -> i64 {
		i64::try_from(self.notes.len()).unwrap_or(i64::MAX)
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		self.waveform(index)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		let x = self.get_item(index)?;
		let y = matrix::slice(&self.labels, 0, index, index + 1)?;
		Ok(Sample::new_with_label(x, y))
	}
}

/// Read `extracted/index.csv` rows
/// (`note,family,pitch,velocity,instrument,samples`).
fn read_index(extracted: &std::path::Path) -> Result<Vec<Note>> {
	let index = std::fs::read(extracted.join("index.csv"))
		.map_err(|error| Error::io("read nsynth index", error))?;
	let rows = crate::core::csv::parse("nsynth", &index)?;
	let mut notes = Vec::with_capacity(rows.len());
	for (row, record) in rows.iter().enumerate() {
		if record.len() != 6 {
			return Err(Error::data_loss(format!(
				"nsynth index row has {} fields, expected 6",
				record.len()
			)));
		}
		let field = |col: usize| -> Result<f32> {
			record[col]
				.parse::<f32>()
				.map_err(|_| Error::data_loss(format!("nsynth index row {row} field {col} is not numeric")))
		};
		notes.push(Note {
			note_str: record[0].clone(),
			family: field(1)?,
			pitch: field(2)?,
			velocity: field(3)?,
			instrument: field(4)?,
			samples: field(5)?,
		});
	}
	Ok(notes)
}

/// Build the decoded matrix payload for `notes`.
fn decoded(notes: Vec<Note>) -> Decoded {
	let len = notes.len();
	Decoded {
		matrices: vec![
			MatrixData {
				name: "labels",
				shape: vec![len],
				values: notes.iter().map(|note| note.family).collect(),
			},
			MatrixData {
				name: "pitches",
				shape: vec![len],
				values: notes.iter().map(|note| note.pitch).collect(),
			},
			MatrixData {
				name: "velocities",
				shape: vec![len],
				values: notes.iter().map(|note| note.velocity).collect(),
			},
			MatrixData {
				name: "instruments",
				shape: vec![len],
				values: notes.iter().map(|note| note.instrument).collect(),
			},
			MatrixData {
				name: "lengths",
				shape: vec![len],
				values: notes.iter().map(|note| note.samples).collect(),
			},
		],
		text: None,
		seq_len: 0,
	}
}

/// Materialize `extracted/index.csv` and `extracted/wav/*.wav` from
/// the verified tar.gz. Idempotent: the caller only enters when
/// `index.csv` is absent.
fn extract_archive(dir: &std::path::Path, extracted: &std::path::Path) -> Result<()> {
	let staging = extracted.with_extension("partial");
	if staging.exists() {
		std::fs::remove_dir_all(&staging)
			.map_err(|error| Error::io("clear nsynth partial extract", error))?;
	}
	std::fs::create_dir_all(staging.join("wav"))
		.map_err(|error| Error::io("create nsynth wav dir", error))?;
	let tar_path = dir.join("nsynth-test.jsonwav.tar.gz");

	// Pass 1: `examples.json` note metadata keyed by note_str.
	let mut meta: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
	let mut found_json = false;
	extract::walk_tar_gz(&tar_path, |name, data| {
		if !name.ends_with("examples.json") {
			return Ok(());
		}
		let table: serde_json::Map<String, serde_json::Value> =
			serde_json::from_slice(data).map_err(|error| {
				Error::data_loss(format!("nsynth examples.json is not valid JSON: {error}"))
			})?;
		meta = table;
		found_json = true;
		Ok(())
	})?;
	if !found_json {
		return Err(Error::data_loss("nsynth archive lacks examples.json"));
	}

	// Pass 2: land clips and join index rows.
	let mut index = String::new();
	extract::walk_tar_gz(&tar_path, |name, data| {
		let name = name.strip_prefix("./").unwrap_or(name);
		let Some(stem) = name.strip_prefix("nsynth-test/audio/") else {
			return Ok(());
		};
		if !stem.ends_with(".wav") {
			return Ok(());
		}
		let stem = stem.strip_suffix(".wav").unwrap_or(stem);
		let view = wav::parse(data)?;
		if view.sample_rate != SAMPLE_RATE || view.channels != 1 || view.bits_per_sample != 16 {
			return Err(Error::data_loss(format!(
				"nsynth note {stem} is {} Hz x {} x {}-bit, expected {SAMPLE_RATE} Hz mono PCM16",
				view.sample_rate, view.channels, view.bits_per_sample
			)));
		}
		let record = meta
			.get(stem)
			.ok_or_else(|| Error::data_loss(format!("nsynth note {stem} has no examples.json row")))?;
		let number = |key: &str| -> Result<f64> {
			record[key]
				.as_f64()
				.ok_or_else(|| Error::data_loss(format!("nsynth note {stem} {key} is not numeric")))
		};
		index.push_str(&format!(
			"{stem},{},{},{},{},{}\n",
			number("instrument_family")?,
			number("pitch")?,
			number("velocity")?,
			number("instrument")?,
			view.pcm.len() / 2
		));
		std::fs::write(staging.join("wav").join(format!("{stem}.wav")), data)
			.map_err(|error| Error::io("write nsynth clip", error))?;
		Ok(())
	})?;

	std::fs::write(staging.join("index.csv"), index)
		.map_err(|error| Error::io("write nsynth index", error))?;
	if extracted.exists() {
		std::fs::remove_dir_all(extracted)
			.map_err(|error| Error::io("replace nsynth extract", error))?;
	}
	std::fs::rename(&staging, extracted)
		.map_err(|error| Error::io("publish nsynth extract", error))?;
	Ok(())
}
