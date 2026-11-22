//! CREMA-D speech-emotion clips as a device-resident [`oa::data::Dataset`].
//!
//! `crema-d-mirror-main.tar.gz` is materialized through
//! [`ensure_pack`](crate::pack::ensure_pack) under `crema_d/1.0.0`; the
//! `AudioWAV/*.wav` members decode to a `[N, 1, T]` F32 waveform matrix
//! zero-padded to the longest clip, a `[N]` F32 matrix of true sample
//! counts, and `[N]` F32 emotion indices into [`CLASSES`]. Filenames carry
//! `<actor>_<sentence>_<emotion>_<level>.wav`.
//!
//! The corpus ships no official split; [`CremaDSplit`] is a deterministic
//! speaker-disjoint partition derived from sorted actor ids — actors at
//! positions `i % 10 == 0` form test, `i % 10 == 5` validation, and the
//! rest train — so no actor crosses splits. Per-sample `[1, T]` slices
//! collate to `[B, 1, T]` batches through [`oa::data`].

use std::collections::BTreeMap;

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::decode::{Decoded, MatrixData};
use crate::extract;
use crate::pack::{self, FetchProgress};
use crate::wav;

#[path = "crema_d.gen.rs"]
mod generated;

pub(crate) use generated::CREMA_D_PACK;
pub use generated::{CLASSES, CremaDConfig, CremaDSplit, SAMPLE_RATE};

/// Device-resident CREMA-D split implementing [`oa::data::Dataset`].
///
/// `waveforms()` is `[N, 1, T]` F32 zero-padded, `lengths()` is `[N]` F32
/// true sample counts at [`SAMPLE_RATE`], and `labels()` is `[N]` F32
/// emotion indices into [`CLASSES`].
pub struct CremaD {
	dataset: MatrixDataset,
	lengths: Matrix,
	split: CremaDSplit,
}

impl CremaD {
	/// Fetch, verify, and decode the pack on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, gzip/tar decode errors, WAV
	/// decode errors, and an error when a filename does not carry an actor,
	/// sentence, emotion, and level field.
	pub fn decode(config: &CremaDConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &CremaDConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&CREMA_D_PACK, &root, progress)?;

		let mut clips: Vec<(u32, f32, Vec<f32>)> = Vec::new();
		extract::walk_tar_gz(&dir.join("crema-d-mirror-main.tar.gz"), |name, data| {
			let Some(file) = name.strip_prefix("crema-d-mirror-main/AudioWAV/") else {
				return Ok(());
			};
			if !file.ends_with(".wav") {
				return Ok(());
			}
			let (actor, label) = parse_name(file)?;
			let view = wav::parse(data)?;
			if view.channels != 1 || view.bits_per_sample != 16 || view.sample_rate != SAMPLE_RATE as u32
			{
				return Err(Error::data_loss(format!(
					"crema_d clip {file} is not 16000 Hz 16-bit mono PCM"
				)));
			}
			let mut values = Vec::with_capacity(view.pcm.len() / 2);
			for pair in view.pcm.as_chunks::<2>().0 {
				values.push(f32::from(i16::from_le_bytes(*pair)) / 32768.0);
			}
			clips.push((actor, label, values));
			Ok(())
		})?;
		if clips.is_empty() {
			return Err(Error::data_loss("crema_d pack holds no AudioWAV clips"));
		}

		// Derived speaker-disjoint split: sorted actors, every 10th to test,
		// every 10th offset 5 to validation, rest to train.
		let mut actors: Vec<u32> = clips.iter().map(|(actor, _, _)| *actor).collect();
		actors.sort_unstable();
		actors.dedup();
		let assignment: BTreeMap<u32, CremaDSplit> = actors
			.iter()
			.enumerate()
			.map(|(index, &actor)| {
				let split = if index % 10 == 0 {
					CremaDSplit::Test
				} else if index % 10 == 5 {
					CremaDSplit::Validation
				} else {
					CremaDSplit::Train
				};
				(actor, split)
			})
			.collect();
		clips.retain(|(actor, _, _)| assignment[actor] == config.split);
		if clips.is_empty() {
			return Err(Error::data_loss(format!(
				"crema_d {:?} split holds no clips",
				config.split
			)));
		}

		let max_len = clips.iter().map(|(_, _, v)| v.len()).max().unwrap_or(0);
		let mut waveforms = vec![0.0f32; clips.len() * max_len];
		let mut lengths = Vec::with_capacity(clips.len());
		let mut labels = Vec::with_capacity(clips.len());
		for (row, (_, label, values)) in clips.iter().enumerate() {
			waveforms[row * max_len..row * max_len + values.len()].copy_from_slice(values);
			lengths.push(values.len() as f32);
			labels.push(*label);
		}
		Ok(Decoded {
			matrices: vec![
				MatrixData {
					name: "waveforms",
					shape: vec![clips.len(), 1, max_len],
					values: waveforms,
				},
				MatrixData {
					name: "lengths",
					shape: vec![lengths.len()],
					values: lengths,
				},
				MatrixData {
					name: "labels",
					shape: vec![labels.len()],
					values: labels,
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
		config: &CremaDConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Ok(Self {
			dataset: MatrixDataset::new(
				decoded.upload("waveforms", engine)?,
				decoded.upload("labels", engine)?,
			)?,
			lengths: decoded.upload("lengths", engine)?,
			split: config.split,
		})
	}

	/// Full `[N, 1, T]` F32 zero-padded waveform matrix on the owning
	/// engine.
	#[must_use]
	pub const fn waveforms(&self) -> &Matrix {
		self.dataset.x()
	}

	/// Full `[N]` F32 clip sample counts at [`SAMPLE_RATE`] on the owning
	/// engine.
	#[must_use]
	pub const fn lengths(&self) -> &Matrix {
		&self.lengths
	}

	/// Full `[N]` F32 emotion label matrix on the owning engine; indices
	/// into [`CLASSES`].
	#[must_use]
	pub const fn labels(&self) -> &Matrix {
		self.dataset.y()
	}

	/// Materialized split.
	#[must_use]
	pub const fn split(&self) -> CremaDSplit {
		self.split
	}
}

impl Dataset for CremaD {
	fn len(&self) -> i64 {
		self.dataset.len()
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		self.dataset.get_item(index)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		self.dataset.get_sample(index)
	}
}

/// `<actor>_<sentence>_<emotion>_<level>.wav` -> `(actor_id, emotion_index)`.
fn parse_name(file: &str) -> Result<(u32, f32)> {
	let stem = file.strip_suffix(".wav").unwrap_or(file);
	let mut parts = stem.split('_');
	let actor: u32 = parts
		.next()
		.and_then(|part| part.parse().ok())
		.ok_or_else(|| Error::data_loss(format!("crema_d clip {file} has no actor id")))?;
	let emotion = parts
		.nth(1)
		.ok_or_else(|| Error::data_loss(format!("crema_d clip {file} has no emotion")))?;
	let label = CLASSES
		.iter()
		.position(|&name| name == emotion)
		.map(|index| index as f32)
		.ok_or_else(|| Error::data_loss(format!("crema_d clip {file} emotion is unknown")))?;
	Ok((actor, label))
}
