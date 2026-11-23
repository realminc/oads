//! Street View House Numbers (SVHN) cropped digits as a
//! [`oa::data::Dataset`].
//!
//! `{train,test,extra}_32x32.mat` are materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `svhn/1.0.0` and decode through the minimal MAT 5.0 reader in
//! `crate::core::mat`. Each file carries `X` as a column-major
//! `[32, 32, 3, N]` uint8 array and `y` as `[N, 1]` labels in
//! `1..=10` (10 encodes digit 0); the loader reorders `X` to
//! `[N, 3, 32, 32]` F32 NCHW and remaps labels to `0..9`. `extra`
//! holds 531,131 easier samples for augmentation or
//! semi-supervised training.

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::mat;
use crate::core::pack::{self, FetchProgress};

#[path = "svhn.gen.rs"]
mod generated;

pub(crate) use generated::SVHN_PACK;
pub use generated::{SvhnConfig, SvhnSplit};

fn member(split: SvhnSplit) -> &'static str {
	match split {
		SvhnSplit::Train => "train_32x32.mat",
		SvhnSplit::Extra => "extra_32x32.mat",
		SvhnSplit::Test => "test_32x32.mat",
	}
}

/// Materialized SVHN split on an engine.
pub struct Svhn {
	dataset: MatrixDataset,
	split: SvhnSplit,
}

impl Svhn {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and MAT decode errors
	/// when a member is missing or an array shape is unexpected.
	pub fn decode(config: &SvhnConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled`
	/// when the callback aborts a fetch.
	pub fn decode_with_progress(
		config: &SvhnConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&SVHN_PACK, &root, progress)?;
		let scale = if config.normalize { 1.0 / 255.0 } else { 1.0 };
		let (images, labels) = decode_member(&dir.join(member(config.split)), scale)?;
		let count = labels.len();
		Ok(Decoded {
			matrices: vec![
				MatrixData {
					name: "images",
					shape: vec![count, 3, 32, 32],
					values: images,
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
	/// Returns the errors of [`load_on`](Self::load_on), plus `Cancelled`
	/// when the callback aborts a fetch.
	pub fn load_with_progress_on(
		engine: &Engine,
		config: &SvhnConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Ok(Self {
			dataset: MatrixDataset::new(
				decoded.upload("images", engine)?,
				decoded.upload("labels", engine)?,
			)?,
			split: config.split,
		})
	}

	/// Full `[N, 3, 32, 32]` F32 image matrix on the owning engine.
	#[must_use]
	pub const fn images(&self) -> &Matrix {
		self.dataset.x()
	}

	/// Full `[N]` F32 digit label matrix (`0..9`) on the owning engine.
	#[must_use]
	pub const fn labels(&self) -> &Matrix {
		self.dataset.y()
	}

	/// The split this instance loaded.
	#[must_use]
	pub const fn split(&self) -> SvhnSplit {
		self.split
	}
}

impl Dataset for Svhn {
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

/// Decode one `*_32x32.mat` member to `[N, 3, 32, 32]` F32 NCHW pixels
/// plus `[N]` labels, applying `scale` to pixels.
fn decode_member(path: &std::path::Path, scale: f32) -> Result<(Vec<f32>, Vec<f32>)> {
	let body = std::fs::read(path).map_err(|error| Error::io("read svhn member", error))?;
	let arrays = mat::read_arrays("svhn", &body)?;
	let x = arrays
		.get("X")
		.ok_or_else(|| Error::data_loss("svhn mat lacks X"))?;
	let y = arrays
		.get("y")
		.ok_or_else(|| Error::data_loss("svhn mat lacks y"))?;
	if x.dims.len() != 4 || x.dims[0] != 32 || x.dims[1] != 32 || x.dims[2] != 3 {
		return Err(Error::data_loss(format!(
			"svhn X dims are {:?}, expected [32, 32, 3, N]",
			x.dims
		)));
	}
	let n = x.dims[3];
	if y.values.len() != n {
		return Err(Error::data_loss(format!(
			"svhn y has {} labels for {n} images",
			y.values.len()
		)));
	}
	let mut images = vec![0.0f32; n * 3 * 32 * 32];
	for (sample, chunk) in images.chunks_mut(3 * 32 * 32).enumerate() {
		for c in 0..3 {
			for w in 0..32 {
				for h in 0..32 {
					let src = h + w * 32 + c * 32 * 32 + sample * 3 * 32 * 32;
					chunk[c * 32 * 32 + h * 32 + w] = x.values[src] as f32 * scale;
				}
			}
		}
	}
	let labels = y
		.values
		.iter()
		.map(|&v| {
			let digit = v as u32;
			f32::from((digit % 10) as u8)
		})
		.collect();
	Ok((images, labels))
}
