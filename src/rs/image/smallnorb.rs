//! Small NORB stereo-pair objects as a [`oa::data::Dataset`].
//!
//! `smallnorb-*-{dat,cat,info}.mat.gz` members are materialized
//! through [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `smallnorb/1.0.0`. The files carry NORB's own matrix format
//! (magic `0x1E3D4C5x`, dim count, LE dims, payload), not MATLAB:
//! `dat` is `[N, 2, 96, 96]` uint8 stereo pairs, `cat` is `[N]`
//! int32 class indices (animal/human/airplane/truck/car), and
//! `info` is `[N, 4]` int32 (instance, elevation, azimuth,
//! lighting) metadata. Pixels upload as `[N, 2, 96, 96]` F32.

use std::path::Path;

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Error, Result, Runtime};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::pack::{self, FetchProgress};

#[path = "smallnorb.gen.rs"]
mod generated;

pub(crate) use generated::SMALLNORB_PACK;
pub use generated::{SmallnorbConfig, SmallnorbSplit};

const MAGIC_INT32: u32 = 0x1E3D_4C54;
const MAGIC_UINT8: u32 = 0x1E3D_4C55;

/// Object class names in canonical order; label index `i` names
/// `CLASSES[i]`.
pub const CLASSES: [&str; 5] = ["animal", "human", "airplane", "truck", "car"];

fn member(split: SmallnorbSplit) -> &'static str {
	match split {
		SmallnorbSplit::Train => "training",
		SmallnorbSplit::Test => "testing",
	}
}

/// Materialized small NORB split on an engine.
pub struct Smallnorb {
	dataset: MatrixDataset,
	info: Matrix,
	split: SmallnorbSplit,
}

impl Smallnorb {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and NORB matrix decode
	/// errors when a member is missing or a header is malformed.
	pub fn decode(config: &SmallnorbConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled`
	/// when the callback aborts a fetch.
	pub fn decode_with_progress(
		config: &SmallnorbConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&SMALLNORB_PACK, &root, progress)?;
		let base = member(config.split);
		let scale = if config.normalize { 1.0 / 255.0 } else { 1.0 };
		let (dims, pixels) = read_matrix(&dir.join(format!("{base}-dat.gz")), MAGIC_UINT8)?;
		let (_, labels) = read_matrix(&dir.join(format!("{base}-cat.gz")), MAGIC_INT32)?;
		let (info_dims, info) = read_matrix(&dir.join(format!("{base}-info.gz")), MAGIC_INT32)?;
		if dims.len() != 4 || dims[1] != 2 || dims[2] != 96 || dims[3] != 96 {
			return Err(Error::data_loss(format!(
				"smallnorb dat dims are {dims:?}, expected [N, 2, 96, 96]"
			)));
		}
		if labels.len() != dims[0] || info.len() != dims[0] * 4 {
			return Err(Error::data_loss("smallnorb member counts disagree"));
		}
		let count = dims[0];
		Ok(Decoded {
			matrices: vec![
				MatrixData {
					name: "images",
					shape: dims,
					values: pixels.iter().map(|&v| v * scale).collect(),
				},
				MatrixData {
					name: "labels",
					shape: vec![count],
					values: labels,
				},
				MatrixData {
					name: "info",
					shape: info_dims,
					values: info,
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
		config: &SmallnorbConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Ok(Self {
			dataset: MatrixDataset::new(
				decoded.upload("images", engine)?,
				decoded.upload("labels", engine)?,
			)?,
			info: decoded.upload("info", engine)?,
			split: config.split,
		})
	}

	/// Full `[N, 2, 96, 96]` F32 stereo-pair matrix on the owning
	/// engine.
	#[must_use]
	pub const fn images(&self) -> &Matrix {
		self.dataset.x()
	}

	/// Full `[N]` F32 class label matrix into [`CLASSES`].
	#[must_use]
	pub const fn labels(&self) -> &Matrix {
		self.dataset.y()
	}

	/// Full `[N, 4]` F32 (instance, elevation, azimuth, lighting)
	/// metadata matrix on the owning engine.
	#[must_use]
	pub const fn info(&self) -> &Matrix {
		&self.info
	}

	/// The split this instance loaded.
	#[must_use]
	pub const fn split(&self) -> SmallnorbSplit {
		self.split
	}
}

impl Dataset for Smallnorb {
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

/// Read one NORB matrix file: magic, u32 dim count, LE u32 dims,
/// payload elements widened to F32.
fn read_matrix(path: &Path, magic: u32) -> Result<(Vec<usize>, Vec<f32>)> {
	let body = crate::core::extract::gunzip(path)?;
	if body.len() < 8 {
		return Err(Error::data_loss("smallnorb member is truncated"));
	}
	let le32 = |offset: usize| -> u32 {
		u32::from_le_bytes(body[offset..offset + 4].try_into().expect("slice"))
	};
	if le32(0) != magic {
		return Err(Error::data_loss(format!(
			"smallnorb magic is {:#x}, expected {magic:#x}",
			le32(0)
		)));
	}
	let ndim = le32(4) as usize;
	let head = 8 + ndim * 4;
	if body.len() < head {
		return Err(Error::data_loss("smallnorb header is truncated"));
	}
	let mut count = 1usize;
	let mut dims = Vec::with_capacity(ndim);
	for i in 0..ndim {
		let d = le32(8 + i * 4) as usize;
		count = count
			.checked_mul(d)
			.ok_or_else(|| Error::data_loss("smallnorb dims overflow"))?;
		dims.push(d);
	}
	let payload = &body[head..];
	let values = match magic {
		MAGIC_UINT8 => {
			if payload.len() < count {
				return Err(Error::data_loss("smallnorb dat payload is truncated"));
			}
			payload[..count].iter().map(|&b| f32::from(b)).collect()
		}
		MAGIC_INT32 => {
			if payload.len() < count * 4 {
				return Err(Error::data_loss("smallnorb int32 payload is truncated"));
			}
			payload[..count * 4]
				.as_chunks::<4>()
				.0
				.iter()
				.map(|b| i32::from_le_bytes(*b) as f32)
				.collect()
		}
		_ => return Err(Error::internal("smallnorb reader reached a bad magic")),
	};
	Ok((dims, values))
}
