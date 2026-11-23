//! LibriTTS `dev-clean` as a lazy [`oa::data::Dataset`].
//!
//! `dev-clean.tar.gz` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `libritts/1.0.0`. First load streams the archive twice: once to
//! join `*.normalized.txt` transcripts by utterance id, once to land
//! WAV clips under `extracted/wav/<speaker>/<chapter>/` and write an
//! `extracted/index.csv` joining each clip to its speaker and sample
//! count. Only the `[N]` F32 speaker index, `[N]` length, and
//! `[N, K]` transcript matrices are resident;
//! [`oa::data::Dataset::get_item`] lazily decodes one `[1, T_i]` F32
//! mono waveform through [`oa::audio`].
//!
//! Upstream also ships `*.original.txt` sidecars carrying the raw
//! (un-normalized) transcript; the index stores the normalized form,
//! which is the canonical TTS training target.

use std::collections::HashMap;
use std::path::PathBuf;

use oa::data::{Dataset, Sample};
use oa::matrix::{self, Matrix};
use oa::{Engine, Error, Result};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::pack::{self, FetchProgress};
use crate::core::{extract, wav};

#[path = "libritts.gen.rs"]
mod generated;

pub(crate) use generated::LIBRITTS_PACK;
pub use generated::LibrittsConfig;

const PACK_DIR: &str = "libritts/1.0.0";
const SAMPLE_RATE: u32 = 24_000;

/// LibriTTS `dev-clean`: 5,736 utterances from 40 speakers.
///
/// `get_item` lazily decodes a `[1, T_i]` F32 mono waveform at
/// 24,000 Hz; `speaker_ids()` is the resident `[N]` F32 index matrix
/// into [`speakers`](Self::speakers), `lengths()` the `[N]` sample
/// counts, and `transcripts()` the `[N, K]` padded normalized-text
/// byte matrix.
pub struct Libritts {
	speaker_ids: Matrix,
	lengths: Matrix,
	transcripts: Matrix,
	transcript_lengths: Matrix,
	clips: Vec<(String, String)>,
	speakers: Vec<String>,
	wav_dir: PathBuf,
	engine: Engine,
}

impl Libritts {
	/// Fetch, verify, extract, and decode the pack on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, tar decode errors, WAV
	/// header errors, and I/O errors while materializing the extracted
	/// clip cache.
	pub fn decode(config: &LibrittsConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled`
	/// when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &LibrittsConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&LIBRITTS_PACK, &root, progress)?;
		let extracted = dir.join("extracted");
		if !extracted.join("index.csv").exists() {
			extract_archive(&dir, &extracted)?;
		}

		let speakers = read_speakers(&extracted)?;
		let index = std::fs::read(extracted.join("index.csv"))
			.map_err(|error| Error::io("read libritts index", error))?;
		let rows = crate::core::csv::parse("libritts", &index)?;
		let mut max_text = 0_usize;
		for record in &rows {
			if record.len() != 4 {
				return Err(Error::data_loss(format!(
					"libritts index row has {} fields, expected 4",
					record.len()
				)));
			}
			max_text = max_text.max(record[3].len());
		}
		let mut speaker_ids = Vec::with_capacity(rows.len());
		let mut lengths = Vec::with_capacity(rows.len());
		let mut transcripts = vec![0.0f32; rows.len() * max_text];
		let mut transcript_lengths = Vec::with_capacity(rows.len());
		for (row, record) in rows.iter().enumerate() {
			speaker_ids.push(speaker_index(&speakers, &record[1])?);
			lengths.push(record[2].parse::<f32>().map_err(|_| {
				Error::data_loss(format!(
					"libritts index row {row} sample count is not numeric"
				))
			})?);
			let bytes = record[3].as_bytes();
			transcript_lengths.push(bytes.len() as f32);
			for (col, &byte) in bytes.iter().enumerate() {
				transcripts[row * max_text + col] = f32::from(byte);
			}
		}
		Ok(Decoded {
			matrices: vec![
				MatrixData {
					name: "speaker_ids",
					shape: vec![speaker_ids.len()],
					values: speaker_ids,
				},
				MatrixData {
					name: "lengths",
					shape: vec![lengths.len()],
					values: lengths,
				},
				MatrixData {
					name: "transcripts",
					shape: vec![rows.len(), max_text],
					values: transcripts,
				},
				MatrixData {
					name: "transcript_lengths",
					shape: vec![transcript_lengths.len()],
					values: transcript_lengths,
				},
			],
			text: None,
			seq_len: 0,
		})
	}

	/// [`load_on`](Self::load_on) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`load_on`](Self::load_on), plus `Cancelled`
	/// when the callback aborts a fetch.
	pub fn load_with_progress_on(
		engine: &Engine,
		config: &LibrittsConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let decoded = Self::decode_with_progress(config, progress)?;
		let extracted = root.join(PACK_DIR).join("extracted");
		let index = std::fs::read(extracted.join("index.csv"))
			.map_err(|error| Error::io("read libritts index", error))?;
		let rows = crate::core::csv::parse("libritts", &index)?;
		let clips = rows
			.iter()
			.map(|record| (record[0].clone(), record[1].clone()))
			.collect();
		let speakers = read_speakers(&extracted)?;
		Ok(Self {
			speaker_ids: decoded.upload("speaker_ids", engine)?,
			lengths: decoded.upload("lengths", engine)?,
			transcripts: decoded.upload("transcripts", engine)?,
			transcript_lengths: decoded.upload("transcript_lengths", engine)?,
			clips,
			speakers,
			wav_dir: extracted.join("wav"),
			engine: engine.clone(),
		})
	}

	/// Sorted speaker ids present in the extracted corpus; speaker
	/// label index `i` names `speakers()[i]`.
	#[must_use]
	pub fn speakers(&self) -> &[String] {
		&self.speakers
	}

	/// Full `[N]` F32 speaker index matrix on the owning engine;
	/// indices into [`speakers`](Self::speakers).
	#[must_use]
	pub const fn speaker_ids(&self) -> &Matrix {
		&self.speaker_ids
	}

	/// Full `[N]` F32 clip sample counts at 24,000 Hz on the owning
	/// engine.
	#[must_use]
	pub const fn lengths(&self) -> &Matrix {
		&self.lengths
	}

	/// Full `[N, K]` F32 padded normalized-transcript byte matrix on
	/// the owning engine.
	#[must_use]
	pub const fn transcripts(&self) -> &Matrix {
		&self.transcripts
	}

	/// Full `[N]` F32 transcript byte lengths on the owning engine.
	#[must_use]
	pub const fn transcript_lengths(&self) -> &Matrix {
		&self.transcript_lengths
	}

	/// Utterance id (`<speaker>_<chapter>_<book>_<n>`) and speaker id
	/// of row `index`.
	///
	/// # Errors
	///
	/// Returns `out_of_range` when `index` is outside the dataset.
	pub fn clip_id(&self, index: i64) -> Result<(&str, &str)> {
		if index < 0 || index >= self.len() {
			return Err(Error::out_of_range("libritts index is outside the dataset"));
		}
		let (file, speaker) = &self.clips[index as usize];
		Ok((file, speaker))
	}

	/// Lazily decode clip `index` to a `[1, T_i]` F32 mono waveform on
	/// the owning engine.
	///
	/// # Errors
	///
	/// Returns `out_of_range` for an out-of-range index and the
	/// [`oa::audio::decode_file_on`] errors when the extracted clip is
	/// not 24,000 Hz mono PCM.
	pub fn waveform(&self, index: i64) -> Result<Matrix> {
		let (file, speaker) = self.clip_id(index)?;
		// `file` is `<speaker>_<chapter>_<book>_<n>`; the clip lives
		// under `wav/<speaker>/<chapter>/<file>.wav`.
		let chapter = file
			.split('_')
			.nth(1)
			.ok_or_else(|| Error::internal("libritts clip id lacks a chapter"))?;
		let audio = oa::audio::decode_file_on(
			&self.engine,
			self
				.wav_dir
				.join(speaker)
				.join(chapter)
				.join(format!("{file}.wav")),
		)?;
		if audio.sample_rate() != SAMPLE_RATE || audio.as_matrix().shape()[0] != 1 {
			return Err(Error::data_loss(format!(
				"libritts clip {file} is not {SAMPLE_RATE} Hz mono WAV"
			)));
		}
		Ok(audio.as_matrix().clone())
	}
}

impl Dataset for Libritts {
	fn len(&self) -> i64 {
		i64::try_from(self.clips.len()).unwrap_or(i64::MAX)
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		self.waveform(index)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		let x = self.get_item(index)?;
		let y = matrix::slice(&self.speaker_ids, 0, index, index + 1)?;
		Ok(Sample::new_with_label(x, y))
	}
}

/// Speaker id -> index into `speakers`.
fn speaker_index(speakers: &[String], speaker: &str) -> Result<f32> {
	speakers
		.iter()
		.position(|name| name == speaker)
		.map(|index| index as f32)
		.ok_or_else(|| Error::data_loss(format!("libritts speaker '{speaker}' is unknown")))
}

/// Read `extracted/speakers.txt` (one id per line, sorted).
fn read_speakers(extracted: &std::path::Path) -> Result<Vec<String>> {
	let data = std::fs::read(extracted.join("speakers.txt"))
		.map_err(|error| Error::io("read libritts speaker list", error))?;
	let text = String::from_utf8(data)
		.map_err(|_| Error::data_loss("libritts speaker list is not valid UTF-8"))?;
	Ok(text.lines().map(str::to_owned).collect())
}

/// Materialize `extracted/index.csv`, `speakers.txt`, and
/// `extracted/wav/<speaker>/<chapter>/*.wav` from the verified
/// tar.gz. Idempotent: the caller only enters when `index.csv` is
/// absent.
fn extract_archive(dir: &std::path::Path, extracted: &std::path::Path) -> Result<()> {
	let staging = extracted.with_extension("partial");
	if staging.exists() {
		std::fs::remove_dir_all(&staging)
			.map_err(|error| Error::io("clear libritts partial extract", error))?;
	}
	std::fs::create_dir_all(staging.join("wav"))
		.map_err(|error| Error::io("create libritts wav dir", error))?;
	let tar_path = dir.join("dev-clean.tar.gz");

	// First pass: join `*.normalized.txt` bodies by utterance id.
	let mut texts: HashMap<String, String> = HashMap::new();
	extract::walk_tar_gz(&tar_path, |name, data| {
		let Some(stem) = name.strip_suffix(".normalized.txt") else {
			return Ok(());
		};
		let Some(id) = stem.rsplit('/').next() else {
			return Ok(());
		};
		let text = String::from_utf8_lossy(data).trim().to_owned();
		texts.insert(id.to_owned(), text);
		Ok(())
	})?;

	// Second pass: materialize clips and index rows.
	let mut index = String::new();
	let mut speakers: Vec<String> = Vec::new();
	extract::walk_tar_gz(&tar_path, |name, data| {
		let name = name.strip_prefix("./").unwrap_or(name);
		let Some(rest) = name.strip_prefix("LibriTTS/dev-clean/") else {
			return Ok(());
		};
		if !rest.ends_with(".wav") {
			return Ok(());
		}
		let mut parts = rest.split('/');
		let (Some(speaker), Some(chapter), Some(file)) = (parts.next(), parts.next(), parts.next())
		else {
			return Err(Error::data_loss(format!(
				"libritts member {name} does not match <speaker>/<chapter>/<id>.wav"
			)));
		};
		let stem = file.strip_suffix(".wav").unwrap_or(file);
		let view = wav::parse(data)?;
		if view.sample_rate != SAMPLE_RATE || view.channels != 1 || view.bits_per_sample != 16 {
			return Err(Error::data_loss(format!(
				"libritts clip {name} is {} Hz x {} x {}-bit, expected {SAMPLE_RATE} Hz mono PCM16",
				view.sample_rate, view.channels, view.bits_per_sample
			)));
		}
		let samples = view.pcm.len() / 2;
		let transcript = texts.get(stem).cloned().unwrap_or_default();
		index.push_str(&format!(
			"{stem},{speaker},{samples},\"{}\"\n",
			transcript.replace('"', "\"\"")
		));
		if !speakers.iter().any(|name| name == speaker) {
			speakers.push(speaker.to_owned());
		}
		let dir = staging.join("wav").join(speaker).join(chapter);
		std::fs::create_dir_all(&dir)
			.map_err(|error| Error::io("create libritts chapter dir", error))?;
		std::fs::write(dir.join(file), data)
			.map_err(|error| Error::io("write libritts clip", error))?;
		Ok(())
	})?;

	std::fs::write(staging.join("index.csv"), index)
		.map_err(|error| Error::io("write libritts index", error))?;
	speakers.sort_unstable();
	speakers.dedup();
	std::fs::write(staging.join("speakers.txt"), speakers.join("\n"))
		.map_err(|error| Error::io("write libritts speaker list", error))?;
	if extracted.exists() {
		std::fs::remove_dir_all(extracted)
			.map_err(|error| Error::io("replace libritts extract", error))?;
	}
	std::fs::rename(&staging, extracted)
		.map_err(|error| Error::io("publish libritts extract", error))?;
	Ok(())
}
