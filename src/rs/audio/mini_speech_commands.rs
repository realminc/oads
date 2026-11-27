//! Google mini Speech Commands as a device-resident [`oa::data::Dataset`].
//!
//! The `mini_speech_commands.zip` payload is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `mini_speech_commands/1.0.0`, walked in memory, and decoded on the host
//! into a `[N, 1, 16000]` F32 waveform matrix and a `[N]` F32 label matrix.
//! Every clip is mono s16 PCM at [`CLIP_SAMPLE_RATE`]; clips shorter than
//! [`CLIP_SAMPLES`] are zero-padded and longer clips truncated.
//! [`oa::data::Dataset::get_item`] returns the `[1, 16000]` slice so
//! [`oa::data::collate`] produces `[B, 1, 16000]` batches.
//!
//! Label `i` names [`COMMANDS`]`[i]`; the ordering matches the canonical
//! mini_speech_commands directory order used by OA's speech-commands
//! tutorial, so the same index space drops into `commands_cnn` workloads.
//! The pack ships no official split — shuffle and partition through
//! [`oa::data::Dataset`] subset operations.

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Error, Result, Runtime};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::extract;
use crate::core::pack::{self, FetchProgress};
use crate::core::wav;

/// Command labels in canonical order; label index `i` names `COMMANDS[i]`.
#[path = "mini_speech_commands.gen.rs"]
mod generated;

pub(crate) use generated::MINI_SPEECH_COMMANDS_PACK;
pub use generated::{CLIP_SAMPLE_RATE, CLIP_SAMPLES, COMMANDS, MiniSpeechCommandsConfig};

/// Device-resident mini Speech Commands dataset implementing
/// [`oa::data::Dataset`].
///
/// `samples()` is `[N, 1, 16000]` F32 mono PCM and `labels()` is `[N]` F32
/// class indices `0..=7` into [`COMMANDS`].
pub struct MiniSpeechCommands {
	dataset: MatrixDataset,
}

impl MiniSpeechCommands {
	/// Fetch, verify, and decode the pack on the host.
	///
	/// The archive is materialized under `config.cache_dir` or the resolved
	/// dataset cache root and walked in memory; every member under
	/// `mini_speech_commands/<command>/*.wav` becomes one clip row.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, zip member errors, WAV decode
	/// errors, and an error when a clip's rate, channel count, or
	/// quantization departs from mono 16 kHz s16 PCM.
	pub fn decode(config: &MiniSpeechCommandsConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// the callback aborts a fetch.
	pub fn decode_with_progress(
		config: &MiniSpeechCommandsConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&MINI_SPEECH_COMMANDS_PACK, &root, progress)?;
		let zip = std::fs::read(dir.join("mini_speech_commands.zip"))
			.map_err(|error| Error::io("read speech commands archive", error))?;
		let scale = if config.normalize {
			1.0_f32 / 32_768.0
		} else {
			1.0
		};
		let mut waveforms = Vec::new();
		let mut label_values = Vec::new();
		extract::walk_zip(&zip, |name, data| {
			let Some(rest) = name.strip_prefix("mini_speech_commands/") else {
				return Ok(());
			};
			let Some((label, file)) = rest.split_once('/') else {
				return Ok(());
			};
			if !file.ends_with(".wav") {
				return Ok(());
			}
			let Some(index) = COMMANDS.iter().position(|command| *command == label) else {
				return Err(Error::data_loss(format!(
					"speech commands archive carries unknown label directory '{label}'"
				)));
			};
			let clip = wav::parse(data)?;
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
			Ok(())
		})?;
		let count = label_values.len();
		if count == 0 {
			return Err(Error::data_loss(
				"speech commands archive holds no command clips",
			));
		}
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
		engine: &Runtime,
		config: &MiniSpeechCommandsConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Ok(Self {
			dataset: MatrixDataset::new(
				decoded.upload("samples", engine)?,
				decoded.upload("labels", engine)?,
			)?,
		})
	}

	/// Wrap caller-owned matrices as a speech-commands dataset.
	///
	/// `samples` must be rank-3 `[N, 1, 16000]` F32 and `labels` rank-1 `[N]`
	/// F32 on the same engine.
	///
	/// # Errors
	///
	/// Returns an error for a misshaped waveform matrix, plus the count,
	/// dtype, and engine errors of [`MatrixDataset::new`].
	pub fn new(samples: Matrix, labels: Matrix) -> Result<Self> {
		let shape = samples.shape();
		if shape.len() != 3 || shape[1] != 1 || shape[2] != CLIP_SAMPLES {
			return Err(Error::invalid_argument(format!(
				"mini_speech_commands requires [N, 1, {CLIP_SAMPLES}] samples"
			)));
		}
		Ok(Self {
			dataset: MatrixDataset::new(samples, labels)?,
		})
	}

	/// Full `[N, 1, 16000]` F32 waveform matrix on the owning engine.
	#[must_use]
	pub const fn samples(&self) -> &Matrix {
		self.dataset.x()
	}

	/// Full `[N]` F32 label matrix on the owning engine; indices into
	/// [`COMMANDS`].
	#[must_use]
	pub const fn labels(&self) -> &Matrix {
		self.dataset.y()
	}
}

impl Dataset for MiniSpeechCommands {
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
