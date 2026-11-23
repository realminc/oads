//! VCTK 0.92 multi-speaker English speech corpus as a device-resident
//! [`oa::data::Dataset`].
//!
//! `DS_10283_3443.zip` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `vctk/0.92.0` and its
//! `wav48_silence_trimmed/<speaker>/*_mic{1,2}.flac` members extract once
//! to `<pack>/extracted/` for reuse (the archive's nested
//! `VCTK-Corpus-0.92.zip` member is a redundant copy and is skipped). The
//! supervised pair is a lazily decoded `[1, T_i]` F32 FLAC waveform at
//! [`SAMPLE_RATE`] -> the speaker index into [`Vctk::speakers`]; `txt/`
//! transcripts ride along as a `[N, K]` padded byte matrix with `[N]`
//! byte lengths. Each microphone channel has slightly different
//! utterance coverage, so [`VctkMic`] selects the index; the corpus
//! ships no official split — partition with [`oa::data::random_split`],
//! or group by speaker for speaker-disjoint evaluation.

use std::collections::HashMap;
use std::path::PathBuf;

use oa::data::{Dataset, Sample};
use oa::matrix::{self, Matrix};
use oa::{Engine, Error, Result};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::extract;
use crate::core::pack::{self, FetchProgress};

#[path = "vctk.gen.rs"]
mod generated;

pub(crate) use generated::VCTK_PACK;
pub use generated::{SAMPLE_RATE, VctkConfig};

const PACK_DIR: &str = "vctk/0.92.0";

/// Microphone channel to materialize.
///
/// VCTK records each utterance on a head-mounted mic (`mic1`) and a desk
/// mic (`mic2`); a small number of clips exist for only one channel, so
/// each channel carries its own clip index.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum VctkMic {
	/// Head-mounted microphone (`*_mic1.flac`); the higher-quality channel.
	#[default]
	Mic1,
	/// Desk microphone (`*_mic2.flac`).
	Mic2,
}

impl VctkMic {
	fn suffix(self) -> &'static str {
		match self {
			Self::Mic1 => "mic1",
			Self::Mic2 => "mic2",
		}
	}
}

/// Device-resident VCTK dataset implementing [`oa::data::Dataset`].
///
/// `get_item` lazily decodes a `[1, T_i]` F32 mono waveform;
/// `speaker_ids()` is the resident `[N]` F32 index matrix into
/// [`speakers`](Self::speakers), `lengths()` the `[N]` sample counts, and
/// `transcripts()` the `[N, K]` padded transcript byte matrix.
pub struct Vctk {
	speaker_ids: Matrix,
	lengths: Matrix,
	transcripts: Matrix,
	transcript_lengths: Matrix,
	clips: Vec<(String, String)>,
	speakers: Vec<String>,
	wav_dir: PathBuf,
	mic: VctkMic,
	engine: Engine,
}

impl Vctk {
	/// Fetch, verify, extract, and decode the pack on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, zip decode errors, FLAC
	/// metadata errors, and I/O errors while materializing the extracted
	/// clip cache.
	pub fn decode(config: &VctkConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &VctkConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&VCTK_PACK, &root, progress)?;
		let extracted = dir.join("extracted");
		if !extracted.join("index_mic1.csv").exists() {
			extract_archive(&dir, &extracted)?;
		}

		let speakers = read_speakers(&extracted)?;
		let mut matrices = Vec::new();
		for mic in [VctkMic::Mic1, VctkMic::Mic2] {
			let index = std::fs::read(extracted.join(format!("index_{}.csv", mic.suffix())))
				.map_err(|error| Error::io("read vctk index", error))?;
			let rows = crate::core::csv::parse("vctk", &index)?;
			let mut max_text = 0_usize;
			for record in &rows {
				if record.len() != 4 {
					return Err(Error::data_loss(format!(
						"vctk index row has {} fields, expected 4",
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
					Error::data_loss(format!("vctk index row {row} sample count is not numeric"))
				})?);
				let bytes = record[3].as_bytes();
				transcript_lengths.push(bytes.len() as f32);
				for (col, &byte) in bytes.iter().enumerate() {
					transcripts[row * max_text + col] = f32::from(byte);
				}
			}
			let names: [&'static str; 4] = match mic {
				VctkMic::Mic1 => [
					"speaker_ids",
					"lengths",
					"transcripts",
					"transcript_lengths",
				],
				VctkMic::Mic2 => [
					"speaker_ids_mic2",
					"lengths_mic2",
					"transcripts_mic2",
					"transcript_lengths_mic2",
				],
			};
			matrices.extend([
				MatrixData {
					name: names[0],
					shape: vec![speaker_ids.len()],
					values: speaker_ids,
				},
				MatrixData {
					name: names[1],
					shape: vec![lengths.len()],
					values: lengths,
				},
				MatrixData {
					name: names[2],
					shape: vec![rows.len(), max_text],
					values: transcripts,
				},
				MatrixData {
					name: names[3],
					shape: vec![transcript_lengths.len()],
					values: transcript_lengths,
				},
			]);
		}
		Ok(Decoded {
			matrices,
			text: None,
			seq_len: 0,
		})
	}

	/// [`load_on`](Self::load_on) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`load_on`](Self::load_on), plus `Cancelled` when
	/// the callback aborts a fetch.
	pub fn load_with_progress_on(
		engine: &Engine,
		config: &VctkConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let decoded = Self::decode_with_progress(config, progress)?;
		let extracted = root.join(PACK_DIR).join("extracted");
		let index = std::fs::read(extracted.join(format!("index_{}.csv", config.mic.suffix())))
			.map_err(|error| Error::io("read vctk index", error))?;
		let rows = crate::core::csv::parse("vctk", &index)?;
		let clips = rows
			.iter()
			.map(|record| (record[0].clone(), record[1].clone()))
			.collect();
		let speakers = read_speakers(&extracted)?;
		let suffix = match config.mic {
			VctkMic::Mic1 => "",
			VctkMic::Mic2 => "_mic2",
		};
		let name = |base: &str| format!("{base}{suffix}");
		Ok(Self {
			speaker_ids: decoded.upload(&name("speaker_ids"), engine)?,
			lengths: decoded.upload(&name("lengths"), engine)?,
			transcripts: decoded.upload(&name("transcripts"), engine)?,
			transcript_lengths: decoded.upload(&name("transcript_lengths"), engine)?,
			clips,
			speakers,
			wav_dir: extracted.join("flac"),
			mic: config.mic,
			engine: engine.clone(),
		})
	}

	/// Sorted speaker ids (`pNNN`) present in the extracted corpus; speaker
	/// label index `i` names `speakers()[i]`.
	#[must_use]
	pub fn speakers(&self) -> &[String] {
		&self.speakers
	}

	/// Full `[N]` F32 speaker index matrix on the owning engine; indices
	/// into [`speakers`](Self::speakers).
	#[must_use]
	pub const fn speaker_ids(&self) -> &Matrix {
		&self.speaker_ids
	}

	/// Full `[N]` F32 clip sample counts at [`SAMPLE_RATE`] on the owning
	/// engine.
	#[must_use]
	pub const fn lengths(&self) -> &Matrix {
		&self.lengths
	}

	/// Full `[N, K]` F32 padded transcript byte matrix on the owning
	/// engine.
	#[must_use]
	pub const fn transcripts(&self) -> &Matrix {
		&self.transcripts
	}

	/// Full `[N]` F32 transcript byte lengths on the owning engine.
	#[must_use]
	pub const fn transcript_lengths(&self) -> &Matrix {
		&self.transcript_lengths
	}

	/// Clip file id (`pNNN_NNN`) and speaker id (`pNNN`) of row `index`.
	///
	/// # Errors
	///
	/// Returns `out_of_range` when `index` is outside the dataset.
	pub fn clip_id(&self, index: i64) -> Result<(&str, &str)> {
		if index < 0 || index >= self.len() {
			return Err(Error::out_of_range("vctk index is outside the dataset"));
		}
		let (file, speaker) = &self.clips[index as usize];
		Ok((file, speaker))
	}

	/// Lazily decode clip `index` to a `[1, T_i]` F32 mono waveform on the
	/// owning engine.
	///
	/// # Errors
	///
	/// Returns `out_of_range` for an out-of-range index and the
	/// [`oa::audio::decode_file_on`] errors when the extracted clip is not
	/// [`SAMPLE_RATE`] mono FLAC.
	pub fn waveform(&self, index: i64) -> Result<Matrix> {
		let (file, speaker) = self.clip_id(index)?;
		let audio = oa::audio::decode_file_on(
			&self.engine,
			self
				.wav_dir
				.join(speaker)
				.join(format!("{file}_{}.flac", self.mic.suffix())),
		)?;
		if audio.sample_rate() != SAMPLE_RATE as u32 || audio.as_matrix().shape()[0] != 1 {
			return Err(Error::data_loss(format!(
				"vctk clip {file} is not {SAMPLE_RATE} Hz mono FLAC"
			)));
		}
		Ok(audio.as_matrix().clone())
	}
}

impl Dataset for Vctk {
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

/// `pNNN` speaker id -> index into `speakers`.
fn speaker_index(speakers: &[String], speaker: &str) -> Result<f32> {
	speakers
		.iter()
		.position(|name| name == speaker)
		.map(|index| index as f32)
		.ok_or_else(|| Error::data_loss(format!("vctk speaker '{speaker}' is unknown")))
}

/// Read `extracted/speakers.txt` (one `pNNN` id per line, sorted).
fn read_speakers(extracted: &std::path::Path) -> Result<Vec<String>> {
	let data = std::fs::read(extracted.join("speakers.txt"))
		.map_err(|error| Error::io("read vctk speaker list", error))?;
	let text = String::from_utf8(data)
		.map_err(|_| Error::data_loss("vctk speaker list is not valid UTF-8"))?;
	Ok(text.lines().map(str::to_owned).collect())
}

/// Sample rate, channel count, and total frame count from a FLAC
/// STREAMINFO block (always the first metadata block after `fLaC`).
fn flac_streaminfo(data: &[u8]) -> Result<(u32, u32, u64)> {
	crate::core::flac::streaminfo("vctk", data)
}

/// Materialize `extracted/index_mic{1,2}.csv`, `speakers.txt`, and
/// `extracted/flac/<speaker>/*_micN.flac` from the verified zip. Idempotent:
/// the caller only enters when `index_mic1.csv` is absent.
fn extract_archive(dir: &std::path::Path, extracted: &std::path::Path) -> Result<()> {
	let staging = extracted.with_extension("partial");
	if staging.exists() {
		std::fs::remove_dir_all(&staging)
			.map_err(|error| Error::io("clear vctk partial extract", error))?;
	}
	std::fs::create_dir_all(staging.join("flac"))
		.map_err(|error| Error::io("create vctk flac dir", error))?;
	let zip_path = dir.join("DS_10283_3443.zip");

	// The datashare bundle wraps the corpus as a nested ~11.7 GB member;
	// stream it to disk once, then index inside it. Deleted before publish.
	let corpus = staging.join("corpus.zip");
	extract::extract_zip_member_to_file(&zip_path, "VCTK-Corpus-0.92.zip", &corpus)?;

	// First pass: collect transcripts so index rows can join them.
	let mut texts: HashMap<String, String> = HashMap::new();
	extract::walk_zip_file(
		&corpus,
		|name| name.starts_with("txt/"),
		|name, data| {
			if let Some((speaker, file)) = name[4..].split_once('/') {
				let stem = file.strip_suffix(".txt").unwrap_or(file);
				let text = String::from_utf8_lossy(data).trim().replace(',', " ");
				texts.insert(format!("{speaker}/{stem}"), text);
			}
			Ok(())
		},
	)?;

	// Second pass: index and materialize both microphone channels.
	let mut indexes: HashMap<&'static str, String> = HashMap::new();
	let mut speakers: Vec<String> = Vec::new();
	extract::walk_zip_file(
		&corpus,
		|name| name.starts_with("wav48_silence_trimmed/"),
		|name, data| {
			let Some((speaker, file)) = name["wav48_silence_trimmed/".len()..].split_once('/') else {
				return Ok(());
			};
			let Some(stem) = file.strip_suffix(".flac") else {
				return Ok(());
			};
			// `stem` is `pNNN_NNN_micN`; the file id drops the mic suffix.
			let Some((file_id, mic_name)) = stem.rsplit_once("_mic") else {
				return Ok(());
			};
			let (rate, channels, samples) = flac_streaminfo(data)?;
			if rate != SAMPLE_RATE as u32 || channels != 1 {
				return Err(Error::data_loss(format!(
					"vctk clip {name} is {rate} Hz x {channels}, expected {SAMPLE_RATE} Hz mono"
				)));
			}
			let key = format!("{speaker}/{file_id}");
			let transcript = texts.get(&key).cloned().unwrap_or_default();
			indexes
				.entry(if mic_name == "1" { "mic1" } else { "mic2" })
				.or_default()
				.push_str(&format!("{file_id},{speaker},{samples},{transcript}\n"));
			if !speakers.iter().any(|name| name == speaker) {
				speakers.push(speaker.to_owned());
			}
			let dir = staging.join("flac").join(speaker);
			std::fs::create_dir_all(&dir).map_err(|error| Error::io("create vctk speaker dir", error))?;
			std::fs::write(dir.join(file), data).map_err(|error| Error::io("write vctk clip", error))?;
			Ok(())
		},
	)?;
	for (mic, index) in &indexes {
		std::fs::write(staging.join(format!("index_{mic}.csv")), index)
			.map_err(|error| Error::io("write vctk index", error))?;
	}
	if !indexes.contains_key("mic1") || !indexes.contains_key("mic2") {
		return Err(Error::data_loss(
			"vctk archive is missing a microphone channel",
		));
	}
	speakers.sort_unstable();
	speakers.dedup();
	std::fs::write(staging.join("speakers.txt"), speakers.join("\n"))
		.map_err(|error| Error::io("write vctk speaker list", error))?;
	std::fs::remove_file(&corpus).map_err(|error| Error::io("remove vctk corpus zip", error))?;
	if extracted.exists() {
		std::fs::remove_dir_all(extracted).map_err(|error| Error::io("replace vctk extract", error))?;
	}
	std::fs::rename(&staging, extracted).map_err(|error| Error::io("publish vctk extract", error))?;
	Ok(())
}
