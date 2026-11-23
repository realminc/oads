//! GTZAN genre collection as a lazy [`oa::data::Dataset`].
//!
//! `genres.tar.gz` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `gtzan/1.0.0`. First load streams the archive to land WAV clips
//! under `extracted/wav/<genre>/` and write `extracted/index.csv`
//! (`clip,genre,samples`), skipping AppleDouble `._` members and
//! `.DS_Store` payloads. Only the `[N]` F32 genre index and `[N]`
//! sample-count matrices are resident;
//! [`oa::data::Dataset::get_item`] lazily decodes one `[1, T_i]` F32
//! mono waveform through [`oa::audio`].

use std::path::PathBuf;

use oa::data::{Dataset, Sample};
use oa::matrix::{self, Matrix};
use oa::{Engine, Error, Result};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::pack::{self, FetchProgress};
use crate::core::{extract, wav};

#[path = "gtzan.gen.rs"]
mod generated;

pub(crate) use generated::GTZAN_PACK;
pub use generated::{GtzanConfig, SAMPLE_RATE};

const PACK_DIR: &str = "gtzan/1.0.0";

/// GTZAN genres: 999 30-second clips across 10 genre classes
/// (the corrupt upstream `jazz.00054.wav` member is skipped).
///
/// `get_item` lazily decodes a `[1, T_i]` F32 mono waveform at
/// [`SAMPLE_RATE`] Hz; `genre_ids()` is the resident `[N]` F32 index
/// matrix into [`genres`](Self::genres) and `lengths()` the `[N]`
/// sample counts.
pub struct Gtzan {
	genre_ids: Matrix,
	lengths: Matrix,
	clips: Vec<(String, String)>,
	genres: Vec<String>,
	wav_dir: PathBuf,
	engine: Engine,
}

impl Gtzan {
	/// Fetch, verify, extract, and decode the pack on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, tar decode errors, WAV
	/// header errors, and I/O errors while materializing the extracted
	/// clip cache.
	pub fn decode(config: &GtzanConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled`
	/// when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &GtzanConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&GTZAN_PACK, &root, progress)?;
		let extracted = dir.join("extracted");
		if !extracted.join("index.csv").exists() {
			extract_archive(&dir, &extracted)?;
		}

		let index = std::fs::read(extracted.join("index.csv"))
			.map_err(|error| Error::io("read gtzan index", error))?;
		let rows = crate::core::csv::parse("gtzan", &index)?;
		let mut genre_ids = Vec::with_capacity(rows.len());
		let mut lengths = Vec::with_capacity(rows.len());
		let genres = read_genres(&extracted)?;
		for (row, record) in rows.iter().enumerate() {
			if record.len() != 3 {
				return Err(Error::data_loss(format!(
					"gtzan index row {row} has {} fields, expected 3",
					record.len()
				)));
			}
			genre_ids.push(
				genres
					.iter()
					.position(|name| name == &record[1])
					.map(|index| index as f32)
					.ok_or_else(|| Error::data_loss(format!("gtzan genre '{}' is unknown", record[1])))?,
			);
			lengths.push(record[2].parse::<f32>().map_err(|_| {
				Error::data_loss(format!("gtzan index row {row} sample count is not numeric"))
			})?);
		}
		Ok(Decoded {
			matrices: vec![
				MatrixData {
					name: "genre_ids",
					shape: vec![genre_ids.len()],
					values: genre_ids,
				},
				MatrixData {
					name: "lengths",
					shape: vec![lengths.len()],
					values: lengths,
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
		config: &GtzanConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let decoded = Self::decode_with_progress(config, progress)?;
		let extracted = root.join(PACK_DIR).join("extracted");
		let index = std::fs::read(extracted.join("index.csv"))
			.map_err(|error| Error::io("read gtzan index", error))?;
		let rows = crate::core::csv::parse("gtzan", &index)?;
		let clips = rows
			.iter()
			.map(|record| (record[0].clone(), record[1].clone()))
			.collect();
		Ok(Self {
			genre_ids: decoded.upload("genre_ids", engine)?,
			lengths: decoded.upload("lengths", engine)?,
			clips,
			genres: read_genres(&extracted)?,
			wav_dir: extracted.join("wav"),
			engine: engine.clone(),
		})
	}

	/// Sorted genre names present in the extracted corpus; genre
	/// index `i` names `genres()[i]`.
	#[must_use]
	pub fn genres(&self) -> &[String] {
		&self.genres
	}

	/// Full `[N]` F32 genre index matrix on the owning engine;
	/// indices into [`genres`](Self::genres).
	#[must_use]
	pub const fn genre_ids(&self) -> &Matrix {
		&self.genre_ids
	}

	/// Full `[N]` F32 clip sample counts at [`SAMPLE_RATE`] on the
	/// owning engine.
	#[must_use]
	pub const fn lengths(&self) -> &Matrix {
		&self.lengths
	}

	/// Clip stem (`<genre>.<n>`) and genre name of row `index`.
	///
	/// # Errors
	///
	/// Returns `out_of_range` when `index` is outside the dataset.
	pub fn clip_id(&self, index: i64) -> Result<(&str, &str)> {
		if index < 0 || index >= self.len() {
			return Err(Error::out_of_range("gtzan index is outside the dataset"));
		}
		let (file, genre) = &self.clips[index as usize];
		Ok((file, genre))
	}

	/// Lazily decode clip `index` to a `[1, T_i]` F32 mono waveform on
	/// the owning engine.
	///
	/// # Errors
	///
	/// Returns `out_of_range` for an out-of-range index and the
	/// [`oa::audio::decode_file_on`] errors when the extracted clip is
	/// not [`SAMPLE_RATE`] Hz mono WAV.
	pub fn waveform(&self, index: i64) -> Result<Matrix> {
		let (file, genre) = self.clip_id(index)?;
		let audio = oa::audio::decode_file_on(
			&self.engine,
			self.wav_dir.join(genre).join(format!("{file}.wav")),
		)?;
		if audio.sample_rate() != SAMPLE_RATE as u32 || audio.as_matrix().shape()[0] != 1 {
			return Err(Error::data_loss(format!(
				"gtzan clip {file} is not {SAMPLE_RATE} Hz mono WAV"
			)));
		}
		Ok(audio.as_matrix().clone())
	}
}

impl Dataset for Gtzan {
	fn len(&self) -> i64 {
		i64::try_from(self.clips.len()).unwrap_or(i64::MAX)
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		self.waveform(index)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		let x = self.get_item(index)?;
		let y = matrix::slice(&self.genre_ids, 0, index, index + 1)?;
		Ok(Sample::new_with_label(x, y))
	}
}

/// Read `extracted/genres.txt` (one genre per line, sorted).
fn read_genres(extracted: &std::path::Path) -> Result<Vec<String>> {
	let data = std::fs::read(extracted.join("genres.txt"))
		.map_err(|error| Error::io("read gtzan genre list", error))?;
	let text =
		String::from_utf8(data).map_err(|_| Error::data_loss("gtzan genre list is not valid UTF-8"))?;
	Ok(text.lines().map(str::to_owned).collect())
}

/// Materialize `extracted/index.csv`, `genres.txt`, and
/// `extracted/wav/<genre>/*.wav` from the verified tar.gz. Idempotent:
/// the caller only enters when `index.csv` is absent.
fn extract_archive(dir: &std::path::Path, extracted: &std::path::Path) -> Result<()> {
	let staging = extracted.with_extension("partial");
	if staging.exists() {
		std::fs::remove_dir_all(&staging)
			.map_err(|error| Error::io("clear gtzan partial extract", error))?;
	}
	std::fs::create_dir_all(staging.join("wav"))
		.map_err(|error| Error::io("create gtzan wav dir", error))?;
	let tar_path = dir.join("genres.tar.gz");

	let mut index = String::new();
	let mut genres: Vec<String> = Vec::new();
	extract::walk_tar_gz(&tar_path, |name, data| {
		let Some(rest) = name.strip_prefix("genres/") else {
			return Ok(());
		};
		let mut parts = rest.split('/');
		let (Some(genre), Some(file)) = (parts.next(), parts.next()) else {
			return Ok(());
		};
		if parts.next().is_some()
			|| !file.ends_with(".wav")
			|| file.starts_with('.')
			|| genre.starts_with('.')
		{
			return Ok(());
		}
		let stem = file.strip_suffix(".wav").unwrap_or(file);
		// Upstream carries one corrupt member (`jazz.00054.wav`); skip
		// non-RIFF payloads rather than fail the pack.
		let Ok(parsed) = wav::parse(data) else {
			return Ok(());
		};
		if parsed.sample_rate != SAMPLE_RATE as u32
			|| parsed.channels != 1
			|| parsed.bits_per_sample != 16
		{
			return Err(Error::data_loss(format!(
				"gtzan clip {name} is {} Hz x {} x {}-bit, expected {SAMPLE_RATE} Hz mono PCM16",
				parsed.sample_rate, parsed.channels, parsed.bits_per_sample
			)));
		}
		let samples = parsed.pcm.len() / 2;
		index.push_str(&format!("{stem},{genre},{samples}\n"));
		if !genres.iter().any(|name| name == genre) {
			genres.push(genre.to_owned());
		}
		let dir = staging.join("wav").join(genre);
		std::fs::create_dir_all(&dir).map_err(|error| Error::io("create gtzan genre dir", error))?;
		std::fs::write(dir.join(file), data).map_err(|error| Error::io("write gtzan clip", error))?;
		Ok(())
	})?;

	std::fs::write(staging.join("index.csv"), index)
		.map_err(|error| Error::io("write gtzan index", error))?;
	genres.sort_unstable();
	genres.dedup();
	std::fs::write(staging.join("genres.txt"), genres.join("\n"))
		.map_err(|error| Error::io("write gtzan genre list", error))?;
	if extracted.exists() {
		std::fs::remove_dir_all(extracted)
			.map_err(|error| Error::io("replace gtzan extract", error))?;
	}
	std::fs::rename(&staging, extracted)
		.map_err(|error| Error::io("publish gtzan extract", error))?;
	Ok(())
}
