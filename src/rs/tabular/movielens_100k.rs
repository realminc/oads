//! MovieLens 100k ratings as a device-resident [`oa::data::Dataset`].
//!
//! `ml-100k.zip` is materialized through [`ensure_pack`](crate::core::pack::ensure_pack)
//! under `movielens_100k/0.9.0`; the official `u1.base`/`u1.test`
//! cross-validation members decode to a `[N, 3]` F32 feature matrix of
//! `(user_id, item_id, timestamp)` triples and a `[N]` F32 rating matrix in
//! `1..=5`. Per-sample slices collate to `[B, 3]` batches through
//! [`oa::data`].

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Error, Result, Runtime};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::extract;
use crate::core::idx::IdxSplit;
use crate::core::pack::{self, FetchProgress};
use crate::tabular::ratings;

#[path = "movielens_100k.gen.rs"]
mod generated;

pub(crate) use generated::MOVIELENS_100K_PACK;
pub use generated::{FEATURE_COUNT, Movielens100kConfig, Movielens100kSplit};

/// Device-resident MovieLens 100k split implementing [`oa::data::Dataset`].
///
/// `features()` is `[N, 3]` F32 `(user_id, item_id, timestamp)` and
/// `ratings()` is `[N]` F32 ratings in `1..=5`.
pub struct Movielens100k {
	dataset: MatrixDataset,
	split: Movielens100kSplit,
}

impl Movielens100k {
	/// Fetch, verify, and decode the pack on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, zip member errors, rating
	/// decode errors, and an error when the configured split member is
	/// missing.
	pub fn decode(config: &Movielens100kConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &Movielens100kConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&MOVIELENS_100K_PACK, &root, progress)?;
		let zip = std::fs::read(dir.join("ml-100k.zip"))
			.map_err(|error| Error::io("read movielens_100k archive", error))?;

		let member = match config.split {
			IdxSplit::Train => "ml-100k/u1.base",
			IdxSplit::Test => "ml-100k/u1.test",
			IdxSplit::Validation => {
				return Err(Error::invalid_argument(
					"movielens_100k carries no validation split",
				));
			}
		};
		let mut ratings_bytes = None;
		extract::walk_zip(&zip, |name, data| {
			if name == member {
				ratings_bytes = Some(data.to_vec());
			}
			Ok(())
		})?;
		let bytes = ratings_bytes
			.ok_or_else(|| Error::data_loss(format!("movielens_100k pack is missing {member}")))?;
		let ratings = ratings::parse("movielens_100k", &bytes, "\t")?;
		Ok(decoded(&ratings))
	}

	/// [`load_on`](Self::load_on) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`load_on`](Self::load_on), plus `Cancelled` when
	/// the callback aborts a fetch.
	pub fn load_with_progress_on(
		engine: &Runtime,
		config: &Movielens100kConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Ok(Self {
			dataset: MatrixDataset::new(
				decoded.upload("features", engine)?,
				decoded.upload("ratings", engine)?,
			)?,
			split: config.split,
		})
	}

	/// Full `[N, 3]` F32 `(user_id, item_id, timestamp)` feature matrix on
	/// the owning engine.
	#[must_use]
	pub const fn features(&self) -> &Matrix {
		self.dataset.x()
	}

	/// Full `[N]` F32 rating matrix in `1..=5` on the owning engine.
	#[must_use]
	pub const fn ratings(&self) -> &Matrix {
		self.dataset.y()
	}

	/// Materialized official split.
	#[must_use]
	pub const fn split(&self) -> Movielens100kSplit {
		self.split
	}
}

impl Dataset for Movielens100k {
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

fn decoded(ratings: &[ratings::Rating]) -> Decoded {
	let mut features = Vec::with_capacity(ratings.len() * FEATURE_COUNT);
	let mut labels = Vec::with_capacity(ratings.len());
	for rating in ratings {
		features.extend_from_slice(&[rating.user_id, rating.item_id, rating.timestamp]);
		labels.push(rating.rating);
	}
	Decoded {
		matrices: vec![
			MatrixData {
				name: "features",
				shape: vec![labels.len(), FEATURE_COUNT],
				values: features,
			},
			MatrixData {
				name: "ratings",
				shape: vec![labels.len()],
				values: labels,
			},
		],
		text: None,
		seq_len: 0,
	}
}
