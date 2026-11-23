//! Moving MNIST frame sequences as a device-resident [`oa::data::Dataset`].
//!
//! `mnist_test_seq.npy` is materialized through [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `moving_mnist/1.0.0` and uploaded once as a `[N*T, 1, 64, 64]` F32
//! matrix: 10,000 sequences of 20 grayscale frames laid out sequence-major.
//! A sample is one whole sequence `[T, 1, 64, 64]`; `get_sample` pairs
//! `frames[0..T-1]` with the one-frame-ahead `frames[1..T]`, the standard
//! next-frame-prediction contract. Storage stays rank-4 because
//! `oa::matrix::slice` is rank-one through rank-four; reshape to
//! `[N, T, 1, 64, 64]` when a per-batch dimension is needed.

use oa::data::{Dataset, Sample};
use oa::matrix::{self, Matrix};
use oa::{Engine, Error, Result};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::npy;
use crate::core::pack::{self, FetchProgress};

#[path = "moving_mnist.gen.rs"]
mod generated;

pub(crate) use generated::MOVING_MNIST_PACK;
pub use generated::MovingMnistConfig;

/// Device-resident Moving MNIST sequences implementing [`oa::data::Dataset`].
///
/// `frames()` is `[N, T, 1, 64, 64]` F32 on the owning engine.
pub struct MovingMnist {
	frames: Matrix,
	seq_len: usize,
}

impl MovingMnist {
	/// Fetch, verify, and decode the sequences on the host.
	///
	/// Returns flat `[N*T, 1, 64, 64]` F32 frames laid out sequence-major
	/// plus the sequence length.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and npy decode errors when the
	/// member is truncated, malformed, or not `[T, N, H, W]` u8.
	pub fn decode(config: &MovingMnistConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// the callback aborts a fetch.
	pub fn decode_with_progress(
		config: &MovingMnistConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&MOVING_MNIST_PACK, &root, progress)?;
		let bytes = std::fs::read(dir.join("mnist_test_seq.npy"))
			.map_err(|error| Error::io("read moving_mnist member", error))?;
		let npy = npy::read("moving_mnist", &bytes)?;
		let [time, total, rows, cols] = match npy.shape.as_slice() {
			[t, n, r, c] => [*t, *n, *r, *c],
			_ => {
				return Err(Error::data_loss(format!(
					"moving_mnist npy shape {:?} is not [T, N, H, W]",
					npy.shape
				)));
			}
		};
		let count = config.count.unwrap_or(total).min(total);
		// npy is time-major [T, N, H, W]; transpose to flat [N*T, 1, H, W].
		let frame = rows * cols;
		let scale = if config.normalize { 1.0 / 255.0 } else { 1.0 };
		let mut values = vec![0.0_f32; count * time * frame];
		for (n, sequence) in values.chunks_exact_mut(time * frame).enumerate() {
			for (t, slot) in sequence.chunks_exact_mut(frame).enumerate() {
				let source = &npy.data[t * frame * total + n * frame..];
				for (pixel, &v) in slot.iter_mut().zip(&source[..frame]) {
					*pixel = f32::from(v) * scale;
				}
			}
		}
		Ok(Decoded {
			matrices: vec![MatrixData {
				name: "frames",
				shape: vec![count * time, 1, rows, cols],
				values,
			}],
			text: None,
			seq_len: time,
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
		config: &MovingMnistConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Ok(Self {
			frames: decoded.upload("frames", engine)?,
			seq_len: decoded.seq_len,
		})
	}

	/// Full `[N*T, 1, 64, 64]` F32 frame matrix on the owning engine.
	#[must_use]
	pub const fn frames(&self) -> &Matrix {
		&self.frames
	}

	/// Sequence length in frames.
	#[must_use]
	pub const fn seq_len(&self) -> usize {
		self.seq_len
	}

	fn check_index(&self, index: i64) -> Result<()> {
		if index < 0 || index >= self.len() {
			return Err(Error::out_of_range(
				"moving_mnist index is outside the dataset",
			));
		}
		Ok(())
	}
}

impl Dataset for MovingMnist {
	fn len(&self) -> i64 {
		let seq_len = i64::try_from(self.seq_len).unwrap_or(i64::MAX);
		i64::try_from(self.frames.shape()[0]).unwrap_or(i64::MAX) / seq_len
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		self.check_index(index)?;
		let seq_len = i64::try_from(self.seq_len).unwrap_or(i64::MAX);
		matrix::slice(&self.frames, 0, index * seq_len, (index + 1) * seq_len)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		self.check_index(index)?;
		let seq_len = i64::try_from(self.seq_len).unwrap_or(i64::MAX);
		let start = index * seq_len;
		let x = matrix::slice(&self.frames, 0, start, start + seq_len - 1)?;
		let y = matrix::slice(&self.frames, 0, start + 1, start + seq_len)?;
		Ok(Sample::new_with_label(x, y))
	}
}
