//! Waves yes/no speech corpus as a [`oa::data::Dataset`].
//!
//! `waves_yesno.tar.gz` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `waves_yesno/1.0.0`. Each `<b0>_<b1>_.._<b7>.wav` member is one
//! recording of eight spoken yes/no tokens; the filename bits are the
//! per-token labels (1 = yes). The corpus is tiny, so waveforms
//! materialize as `[N, 1, T]` F32 zero-padded PCM at 8,000 Hz with
//! `[N]` sample counts; `labels()` is `[N, 8]` F32.

use oa::data::{Dataset, Sample};
use oa::matrix::{self, Matrix};
use oa::{Error, Result, Runtime};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::pack::{self, FetchProgress};
use crate::core::{extract, wav};

#[path = "waves_yesno.gen.rs"]
mod generated;

pub(crate) use generated::WAVES_YESNO_PACK;
pub use generated::WavesYesnoConfig;

const SAMPLE_RATE: u32 = 8_000;

/// Waves yes/no: 60 recordings, 8 labels each.
///
/// `waveforms()` is `[N, 1, T]` F32 zero-padded, `lengths()` is `[N]`
/// F32 sample counts, and `labels()` is `[N, 8]` F32 per-token yes
/// indicators. `get_item` returns one `[1, T]` waveform row;
/// `get_sample` pairs it with the `[8]` label row.
pub struct WavesYesno {
	waveforms: Matrix,
	lengths: Matrix,
	labels: Matrix,
}

impl WavesYesno {
	/// Fetch, verify, and decode the pack on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, tar decode errors, and
	/// WAV header errors when a clip is not 8,000 Hz mono PCM16.
	pub fn decode(config: &WavesYesnoConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled`
	/// when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &WavesYesnoConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&WAVES_YESNO_PACK, &root, progress)?;
		let mut clips: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
		let mut names: Vec<String> = Vec::new();
		extract::walk_tar_gz(&dir.join("waves_yesno.tar.gz"), |name, data| {
			let Some(file) = name.strip_prefix("waves_yesno/") else {
				return Ok(());
			};
			if !file.ends_with(".wav") {
				return Ok(());
			}
			let stem = file.strip_suffix(".wav").unwrap_or(file);
			let bits: Vec<f32> = stem
				.split('_')
				.map(|part| {
					part
						.parse::<u8>()
						.map(|b| f32::from(b.min(1)))
						.map_err(|_| Error::data_loss(format!("waves_yesno name {file} holds a non-bit field")))
				})
				.collect::<Result<_>>()?;
			if bits.len() != 8 {
				return Err(Error::data_loss(format!(
					"waves_yesno name {file} carries {} labels, expected 8",
					bits.len()
				)));
			}
			let view = wav::parse(data)?;
			if view.channels != 1 || view.bits_per_sample != 16 || view.sample_rate != SAMPLE_RATE {
				return Err(Error::data_loss(format!(
					"waves_yesno clip {file} is not {SAMPLE_RATE} Hz 16-bit mono PCM"
				)));
			}
			let mut values = Vec::with_capacity(view.pcm.len() / 2);
			for pair in view.pcm.as_chunks::<2>().0 {
				values.push(f32::from(i16::from_le_bytes(*pair)) / 32768.0);
			}
			clips.push((bits, values));
			names.push(stem.to_owned());
			Ok(())
		})?;
		if clips.is_empty() {
			return Err(Error::data_loss("waves_yesno pack holds no wav clips"));
		}
		// Deterministic order by filename bit pattern.
		let mut order: Vec<usize> = (0..clips.len()).collect();
		order.sort_by(|&a, &b| names[a].cmp(&names[b]));
		let clips: Vec<_> = order.iter().map(|&i| clips[i].clone()).collect();

		let max_len = clips.iter().map(|(_, v)| v.len()).max().unwrap_or(0);
		let mut waveforms = vec![0.0f32; clips.len() * max_len];
		let mut lengths = Vec::with_capacity(clips.len());
		let mut labels = Vec::with_capacity(clips.len() * 8);
		for (row, (bits, values)) in clips.iter().enumerate() {
			waveforms[row * max_len..row * max_len + values.len()].copy_from_slice(values);
			lengths.push(values.len() as f32);
			labels.extend_from_slice(bits);
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
					shape: vec![clips.len(), 8],
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
	/// Returns the errors of [`load_on`](Self::load_on), plus `Cancelled`
	/// when the callback aborts a fetch.
	pub fn load_with_progress_on(
		engine: &Runtime,
		config: &WavesYesnoConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Ok(Self {
			waveforms: decoded.upload("waveforms", engine)?,
			lengths: decoded.upload("lengths", engine)?,
			labels: decoded.upload("labels", engine)?,
		})
	}

	/// Full `[N, 1, T]` F32 zero-padded waveform matrix on the owning
	/// engine.
	#[must_use]
	pub const fn waveforms(&self) -> &Matrix {
		&self.waveforms
	}

	/// Full `[N]` F32 per-clip sample counts at 8,000 Hz.
	#[must_use]
	pub const fn lengths(&self) -> &Matrix {
		&self.lengths
	}

	/// Full `[N, 8]` F32 per-token yes indicators on the owning engine.
	#[must_use]
	pub const fn labels(&self) -> &Matrix {
		&self.labels
	}
}

impl Dataset for WavesYesno {
	fn len(&self) -> i64 {
		i64::from(self.waveforms.shape()[0] as i32)
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		let row = matrix::slice(&self.waveforms, 0, index, index + 1)?;
		row.squeeze(0)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		let x = self.get_item(index)?;
		let y = self.labels.slice(0, index, index + 1)?;
		Ok(Sample::new_with_label(x, y))
	}
}
