//! LJSpeech 1.1 single-speaker TTS corpus as a device-resident
//! [`oa::data::Dataset`].
//!
//! `LJSpeech-1.1.tar.bz2` is materialized through
//! [`ensure_pack`](crate::pack::ensure_pack) under `ljspeech/1.1.0` and its
//! `wavs/` members extract once to `<pack>/extracted/wavs/` for reuse. The
//! supervised pair is `metadata.csv` normalized-transcript bytes ->
//! 22,050 Hz mono PCM: [`LjSpeech::get_item`] returns a `[K]` padded
//! transcript window and [`Dataset::get_sample`] pairs it with a lazily
//! decoded `[1, T_i]` F32 waveform. Clip lengths vary, so `collate` on
//! waveform labels only batches equal-length clips; `audio_lengths`
//! carries each clip's sample count for bucketing.

use std::path::PathBuf;

use oa::data::{Dataset, Sample};
use oa::matrix::{self, Matrix};
use oa::{Engine, Error, Result};

use crate::csv;
use crate::decode::{Decoded, MatrixData};
use crate::extract;
use crate::pack::{self, FetchProgress};
use crate::wav;

#[path = "ljspeech.gen.rs"]
mod generated;

pub(crate) use generated::LJSPEECH_PACK;
pub use generated::{LjSpeechConfig, SAMPLE_RATE};

const PACK_DIR: &str = "ljspeech/1.1.0";

/// Device-resident LJSpeech dataset implementing [`oa::data::Dataset`].
///
/// `transcripts()` is `[N, K]` F32 normalized-transcript bytes padded to
/// the longest record; `audio_lengths()` is `[N]` F32 PCM sample counts at
/// [`SAMPLE_RATE`]. Waveforms decode per sample from the extracted cache.
pub struct LjSpeech {
	transcripts: Matrix,
	transcript_lengths: Matrix,
	audio_lengths: Matrix,
	ids: Vec<String>,
	wav_dir: PathBuf,
	engine: Engine,
}

impl LjSpeech {
	/// Fetch, verify, extract, and decode the pack on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, bzip2/tar decode errors,
	/// metadata decode errors, and I/O errors while materializing the
	/// extracted waveform cache.
	pub fn decode(config: &LjSpeechConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &LjSpeechConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&LJSPEECH_PACK, &root, progress)?;
		let extracted = dir.join("extracted");
		if !extracted.join("lengths.csv").exists() {
			extract_archive(&dir, &extracted)?;
		}

		let metadata = std::fs::read(extracted.join("metadata.csv"))
			.map_err(|error| Error::io("read ljspeech metadata", error))?;
		let records = parse_metadata(&metadata)?;
		let lengths_csv = std::fs::read(extracted.join("lengths.csv"))
			.map_err(|error| Error::io("read ljspeech lengths", error))?;
		let lengths = csv::parse("ljspeech", &lengths_csv)?;

		let mut max_text = 0_usize;
		for (_, normalized) in &records {
			max_text = max_text.max(normalized.len());
		}
		let mut transcripts = vec![0.0f32; records.len() * max_text];
		let mut transcript_lengths = Vec::with_capacity(records.len());
		for (row, (_, normalized)) in records.iter().enumerate() {
			let bytes = normalized.as_bytes();
			transcript_lengths.push(bytes.len() as f32);
			for (col, &byte) in bytes.iter().enumerate() {
				transcripts[row * max_text + col] = f32::from(byte);
			}
		}
		// lengths.csv is written in tar order; align it to metadata order
		// by clip id.
		let mut sample_counts = std::collections::HashMap::new();
		for record in &lengths {
			if record.len() != 2 {
				return Err(Error::data_loss("ljspeech lengths row is malformed"));
			}
			let count = record[1]
				.parse::<f32>()
				.map_err(|_| Error::data_loss(format!("ljspeech length '{}' is not numeric", record[1])))?;
			sample_counts.insert(record[0].clone(), count);
		}
		let mut audio_lengths = Vec::with_capacity(records.len());
		for (id, _) in &records {
			audio_lengths.push(
				*sample_counts
					.get(id)
					.ok_or_else(|| Error::data_loss(format!("ljspeech clip {id} has no recorded length")))?,
			);
		}
		Ok(Decoded {
			matrices: vec![
				MatrixData {
					name: "transcripts",
					shape: vec![records.len(), max_text],
					values: transcripts,
				},
				MatrixData {
					name: "transcript_lengths",
					shape: vec![records.len()],
					values: transcript_lengths,
				},
				MatrixData {
					name: "audio_lengths",
					shape: vec![audio_lengths.len()],
					values: audio_lengths,
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
	/// Returns the errors of [`load_on`](Self::load_on), plus `Cancelled` when
	/// the callback aborts a fetch.
	pub fn load_with_progress_on(
		engine: &Engine,
		config: &LjSpeechConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let decoded = Self::decode_with_progress(config, progress)?;
		let extracted = root.join(PACK_DIR).join("extracted");

		// metadata.csv ids order rows; keep them beside the file list so
		// lazy waveform decode resolves the right clip.
		let metadata = std::fs::read(extracted.join("metadata.csv"))
			.map_err(|error| Error::io("read ljspeech metadata", error))?;
		let records = parse_metadata(&metadata)?;
		let ids = records.iter().map(|(id, _)| id.clone()).collect();
		Ok(Self {
			transcripts: decoded.upload("transcripts", engine)?,
			transcript_lengths: decoded.upload("transcript_lengths", engine)?,
			audio_lengths: decoded.upload("audio_lengths", engine)?,
			ids,
			wav_dir: extracted.join("wavs"),
			engine: engine.clone(),
		})
	}

	/// Full `[N, K]` F32 normalized-transcript byte matrix on the owning
	/// engine, zero-padded to the longest record.
	#[must_use]
	pub const fn transcripts(&self) -> &Matrix {
		&self.transcripts
	}

	/// Full `[N]` F32 transcript byte lengths on the owning engine.
	#[must_use]
	pub const fn transcript_lengths(&self) -> &Matrix {
		&self.transcript_lengths
	}

	/// Full `[N]` F32 waveform sample counts at [`SAMPLE_RATE`] on the
	/// owning engine.
	#[must_use]
	pub const fn audio_lengths(&self) -> &Matrix {
		&self.audio_lengths
	}

	/// Clip id (`LJnnn-nnnn`) of row `index`.
	///
	/// # Errors
	///
	/// Returns `out_of_range` when `index` is outside the dataset.
	pub fn clip_id(&self, index: i64) -> Result<&str> {
		if index < 0 || index >= self.len() {
			return Err(Error::out_of_range("ljspeech index is outside the dataset"));
		}
		Ok(&self.ids[index as usize])
	}

	/// Lazily decode clip `index` to a `[1, T_i]` F32 mono waveform on the
	/// owning engine.
	///
	/// # Errors
	///
	/// Returns `out_of_range` for an out-of-range index and the
	/// [`oa::audio::decode_file_on`] errors when the extracted clip is not
	/// 22,050 Hz mono PCM.
	pub fn waveform(&self, index: i64) -> Result<Matrix> {
		let id = self.clip_id(index)?;
		let audio = oa::audio::decode_file_on(&self.engine, self.wav_dir.join(format!("{id}.wav")))?;
		if audio.sample_rate() != SAMPLE_RATE as u32 || audio.as_matrix().shape()[0] != 1 {
			return Err(Error::data_loss(format!(
				"ljspeech clip {id} is not 22050 Hz mono PCM"
			)));
		}
		Ok(audio.as_matrix().clone())
	}
}

impl Dataset for LjSpeech {
	fn len(&self) -> i64 {
		i64::try_from(self.ids.len()).unwrap_or(i64::MAX)
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		if index < 0 || index >= self.len() {
			return Err(Error::out_of_range("ljspeech index is outside the dataset"));
		}
		matrix::slice(&self.transcripts, 0, index, index + 1)?.squeeze(0)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		let x = self.get_item(index)?;
		let y = self.waveform(index)?;
		Ok(Sample::new_with_label(x, y))
	}
}

/// Parse `id|transcript|normalized_transcript` lines into `(id, normalized)`
/// pairs; transcript text may itself contain `|` or `,` so the id is the
/// first field and the normalized transcript the last.
fn parse_metadata(bytes: &[u8]) -> Result<Vec<(String, String)>> {
	let text = std::str::from_utf8(bytes)
		.map_err(|_| Error::data_loss("ljspeech metadata is not valid UTF-8"))?;
	let mut records = Vec::new();
	for (index, line) in text.lines().enumerate() {
		if line.is_empty() {
			continue;
		}
		let mut parts = line.split('|');
		let id = parts
			.next()
			.ok_or_else(|| Error::data_loss(format!("ljspeech metadata line {} has no id", index + 1)))?;
		let normalized = parts.next_back().ok_or_else(|| {
			Error::data_loss(format!(
				"ljspeech metadata line {} has no normalized transcript",
				index + 1
			))
		})?;
		records.push((id.to_string(), normalized.to_string()));
	}
	Ok(records)
}

/// Materialize `extracted/metadata.csv`, `extracted/lengths.csv`, and
/// `extracted/wavs/*.wav` from the verified bz2 tarball. Idempotent: the
/// caller only enters when `lengths.csv` is absent.
fn extract_archive(dir: &std::path::Path, extracted: &std::path::Path) -> Result<()> {
	let staging = extracted.with_extension("partial");
	if staging.exists() {
		std::fs::remove_dir_all(&staging)
			.map_err(|error| Error::io("clear ljspeech partial extract", error))?;
	}
	let wav_dir = staging.join("wavs");
	std::fs::create_dir_all(&wav_dir).map_err(|error| Error::io("create ljspeech wav dir", error))?;

	let mut lengths = String::new();
	extract::walk_tar_bz2(&dir.join("LJSpeech-1.1.tar.bz2"), |name, data| {
		if name == "LJSpeech-1.1/metadata.csv" {
			std::fs::write(staging.join("metadata.csv"), data)
				.map_err(|error| Error::io("write ljspeech metadata", error))?;
		} else if let Some(file) = name
			.strip_prefix("LJSpeech-1.1/wavs/")
			.filter(|file| file.ends_with(".wav"))
		{
			let view = wav::parse(data)?;
			lengths.push_str(&format!(
				"{},{}\n",
				file.strip_suffix(".wav").unwrap_or(file),
				view.pcm.len() / 2
			));
			std::fs::write(wav_dir.join(file), data)
				.map_err(|error| Error::io("write ljspeech clip", error))?;
		}
		Ok(())
	})?;
	std::fs::write(staging.join("lengths.csv"), lengths)
		.map_err(|error| Error::io("write ljspeech lengths", error))?;
	if extracted.exists() {
		std::fs::remove_dir_all(extracted)
			.map_err(|error| Error::io("replace ljspeech extract", error))?;
	}
	std::fs::rename(&staging, extracted)
		.map_err(|error| Error::io("publish ljspeech extract", error))?;
	Ok(())
}
