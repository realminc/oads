//! Google Speech Commands v0.02 as a device-resident [`oa::data::Dataset`].
//!
//! The 2.4 GB tarball is materialized through
//! [`ensure_pack`](crate::pack::ensure_pack) under `speech_commands/0.2.0`
//! and walked in memory; every member under `<command>/*.wav` becomes one
//! clip row. Members named by `testing_list.txt`/`validation_list.txt` form
//! the test and validation splits; the rest are train. `_background_noise_`
//! and metadata members are not part of the dataset contract.

use std::collections::BTreeMap;

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::decode::{Decoded, MatrixData};
use crate::extract;
use crate::pack::{self, FetchProgress};
use crate::wav;

#[path = "speech_commands.gen.rs"]
mod generated;

pub(crate) use generated::SPEECH_COMMANDS_PACK;
pub use generated::{
	CLIP_SAMPLE_RATE, CLIP_SAMPLES, COMMANDS, SpeechCommandsConfig, SpeechCommandsSplit,
};

/// Device-resident Speech Commands v0.02 split implementing
/// [`oa::data::Dataset`].
///
/// `samples()` is `[N, 1, 16000]` F32 mono PCM and `labels()` is `[N]` F32
/// class indices into [`COMMANDS`].
pub struct SpeechCommands {
	dataset: MatrixDataset,
	split: SpeechCommandsSplit,
}

fn parse_list(text: &str) -> Vec<&str> {
	text
		.lines()
		.map(str::trim)
		.filter(|line| !line.is_empty())
		.collect()
}

impl SpeechCommands {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, archive decode errors, WAV
	/// decode errors, and an error when a clip's rate, channel count, or
	/// quantization departs from mono 16 kHz s16 PCM.
	pub fn decode(config: &SpeechCommandsConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// the callback aborts a fetch.
	pub fn decode_with_progress(
		config: &SpeechCommandsConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&SPEECH_COMMANDS_PACK, &root, progress)?;
		let tar = extract::gunzip(&dir.join("speech_commands_v0.02.tar.gz"))?;
		let test_list = std::str::from_utf8(extract::tar_member(&tar, "testing_list.txt")?)
			.map_err(|_| Error::data_loss("testing_list.txt is not valid UTF-8"))?;
		let validation_list = std::str::from_utf8(extract::tar_member(&tar, "validation_list.txt")?)
			.map_err(|_| Error::data_loss("validation_list.txt is not valid UTF-8"))?;
		let (test_set, validation_set): (Vec<&str>, Vec<&str>) =
			(parse_list(test_list), parse_list(validation_list));
		let wanted = |name: &str| -> Option<()> {
			match config.split {
				SpeechCommandsSplit::Test if test_set.contains(&name) => Some(()),
				SpeechCommandsSplit::Validation if validation_set.contains(&name) => Some(()),
				SpeechCommandsSplit::Train
					if !test_set.contains(&name) && !validation_set.contains(&name) =>
				{
					Some(())
				}
				_ => None,
			}
		};
		let scale = if config.normalize {
			1.0_f32 / 32_768.0
		} else {
			1.0
		};
		let mut members = BTreeMap::new();
		extract::walk_tar(&tar, |name, bytes| {
			let name = name.strip_prefix("./").unwrap_or(name);
			if wanted(name).is_some() {
				members.insert(name.to_owned(), bytes.to_vec());
			}
			Ok(())
		})?;
		let mut waveforms = Vec::new();
		let mut label_values = Vec::new();
		for (name, data) in members {
			let Some((label, file)) = name.split_once('/') else {
				continue;
			};
			if !file.ends_with(".wav") {
				continue;
			}
			let Some(index) = COMMANDS.iter().position(|command| *command == label) else {
				continue;
			};
			let clip = wav::parse(&data)?;
			if clip.channels != 1 || clip.sample_rate != CLIP_SAMPLE_RATE || clip.bits_per_sample != 16 {
				return Err(Error::data_loss(format!(
					"{name}: expected mono {CLIP_SAMPLE_RATE} Hz s16 wav"
				)));
			}
			let base = waveforms.len();
			for &[lo, hi] in clip.pcm.as_chunks::<2>().0.iter().take(CLIP_SAMPLES) {
				waveforms.push(i16::from_le_bytes([lo, hi]) as f32 * scale);
			}
			waveforms.resize(base + CLIP_SAMPLES, 0.0);
			label_values.push(index as f32);
		}
		let count = label_values.len();
		Ok(Decoded {
			matrices: vec![
				MatrixData {
					name: "samples",
					shape: vec![count, 1, CLIP_SAMPLES],
					values: waveforms,
				},
				MatrixData {
					name: "labels",
					shape: vec![count],
					values: label_values,
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
	/// Returns the errors of [`load_on`](Self::load_on), plus `Cancelled` when the
	/// callback aborts a fetch.
	pub fn load_with_progress_on(
		engine: &Engine,
		config: &SpeechCommandsConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Ok(Self {
			dataset: MatrixDataset::new(
				decoded.upload("samples", engine)?,
				decoded.upload("labels", engine)?,
			)?,
			split: config.split,
		})
	}

	/// Full `[N, 1, 16000]` F32 waveform matrix on the owning engine.
	pub const fn samples(&self) -> &Matrix {
		self.dataset.x()
	}

	/// Full `[N]` F32 label matrix on the owning engine.
	pub const fn labels(&self) -> &Matrix {
		self.dataset.y()
	}

	/// Selected split.
	pub const fn split(&self) -> SpeechCommandsSplit {
		self.split
	}
}

impl Dataset for SpeechCommands {
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
