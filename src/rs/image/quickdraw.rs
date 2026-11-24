//! Quick, Draw! 28×28 grayscale bitmap corpus as a device-resident
//! [`oa::data::Dataset`].
//!
//! One `<category>.npy` file per class materializes through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `quickdraw/1.0.0` — 345 `[M, 784]` u8 files in
//! `CATEGORIES` order. `classes` selects the
//! first K categories and `count` caps total decoded samples; with no
//! caps the full pack is ~40M rows (multi-GiB F32), so both knobs are
//! the intended entry point. `images()` is `[N, 1, 28, 28]` F32 and
//! `labels()` is `[N]` F32 category indices.

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::npy;
use crate::core::pack::{self, FetchProgress};

#[path = "quickdraw.gen.rs"]
mod generated;

pub use generated::QuickdrawConfig;
pub(crate) use generated::{CATEGORIES, QUICKDRAW_PACK};

/// Device-resident QuickDraw subset implementing [`oa::data::Dataset`].
///
/// `images()` is `[N, 1, 28, 28]` F32 and `labels()` is `[N]` F32
/// category indices into `CATEGORIES`.
pub struct Quickdraw {
	dataset: MatrixDataset,
	classes: usize,
}

impl Quickdraw {
	/// Fetch, verify, and decode the selected classes on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, npy header errors, and
	/// an error when a class array is not `[M, 784]` u8.
	pub fn decode(config: &QuickdrawConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled`
	/// when `progress` aborts a fetch.
	pub fn decode_with_progress(
		config: &QuickdrawConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&QUICKDRAW_PACK, &root, progress)?;
		let classes = config
			.classes
			.unwrap_or(CATEGORIES.len())
			.min(CATEGORIES.len());
		let scale = if config.normalize { 1.0 / 255.0 } else { 1.0 };
		let mut values: Vec<f32> = Vec::new();
		let mut labels: Vec<f32> = Vec::new();
		for (index, name) in CATEGORIES.iter().enumerate().take(classes) {
			let bytes = std::fs::read(dir.join(format!("{name}.npy")))
				.map_err(|error| Error::io("read quickdraw class member", error))?;
			let array = npy::read("quickdraw", &bytes)?;
			if array.shape.len() != 2 || array.shape[1] != 784 {
				return Err(Error::data_loss(format!(
					"quickdraw class '{name}' has shape {:?}, expected [M, 784]",
					array.shape
				)));
			}
			let remaining = config.count.map(|cap| cap.saturating_sub(labels.len()));
			if matches!(remaining, Some(0)) {
				break;
			}
			let rows = remaining.map_or(array.shape[0], |cap| cap.min(array.shape[0]));
			values.extend(
				array.data[..rows * 784]
					.iter()
					.map(|&byte| f32::from(byte) * scale),
			);
			labels.resize(labels.len() + rows, index as f32);
		}
		let count = labels.len();
		Ok(Decoded {
			matrices: vec![
				MatrixData {
					name: "images",
					shape: vec![count, 1, 28, 28],
					values,
				},
				MatrixData {
					name: "labels",
					shape: vec![count],
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
		config: &QuickdrawConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Ok(Self {
			dataset: MatrixDataset::new(
				decoded.upload("images", engine)?,
				decoded.upload("labels", engine)?,
			)?,
			classes: config
				.classes
				.unwrap_or(CATEGORIES.len())
				.min(CATEGORIES.len()),
		})
	}

	/// Full `[N, 1, 28, 28]` F32 bitmap matrix on the owning engine.
	#[must_use]
	pub const fn images(&self) -> &Matrix {
		self.dataset.x()
	}

	/// Full `[N]` F32 category label matrix on the owning engine.
	#[must_use]
	pub const fn labels(&self) -> &Matrix {
		self.dataset.y()
	}

	/// Number of leading `CATEGORIES` decoded.
	#[must_use]
	pub const fn classes(&self) -> usize {
		self.classes
	}
}

impl Dataset for Quickdraw {
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
