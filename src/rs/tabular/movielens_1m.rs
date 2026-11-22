//! MovieLens 1M ratings as a device-resident [`oa::data::Dataset`].
//!
//! `ml-1m.zip` is materialized through [`ensure_pack`](crate::pack::ensure_pack)
//! under `movielens_1m/0.9.0`; the `ratings.dat` member decodes to a
//! `[N, 3]` F32 feature matrix of `(user_id, item_id, timestamp)` triples
//! and a `[N]` F32 rating matrix in `1..=5`. The pack carries all
//! 1,000,209 ratings unsplit — partition with [`oa::data::random_split`].
//! Per-sample slices collate to `[B, 3]` batches through [`oa::data`].

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::decode::{Decoded, MatrixData};
use crate::extract;
use crate::pack::{self, FetchProgress};
use crate::tabular::ratings;

#[path = "movielens_1m.gen.rs"]
mod generated;

pub(crate) use generated::MOVIELENS_1M_PACK;
pub use generated::{FEATURE_COUNT, Movielens1mConfig};

/// Device-resident MovieLens 1M dataset implementing [`oa::data::Dataset`].
///
/// `features()` is `[N, 3]` F32 `(user_id, item_id, timestamp)` and
/// `ratings()` is `[N]` F32 ratings in `1..=5`.
pub struct Movielens1m {
	dataset: MatrixDataset,
}

impl Movielens1m {
	/// Fetch, verify, and decode the pack on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, zip member errors, rating
	/// decode errors, and an error when `ratings.dat` is missing.
	pub fn decode(config: &Movielens1mConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &Movielens1mConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&MOVIELENS_1M_PACK, &root, progress)?;
		let zip = std::fs::read(dir.join("ml-1m.zip"))
			.map_err(|error| Error::io("read movielens_1m archive", error))?;

		let mut ratings_bytes = None;
		extract::walk_zip(&zip, |name, data| {
			if name == "ml-1m/ratings.dat" {
				ratings_bytes = Some(data.to_vec());
			}
			Ok(())
		})?;
		let bytes = ratings_bytes
			.ok_or_else(|| Error::data_loss("movielens_1m pack is missing ml-1m/ratings.dat"))?;
		let ratings = ratings::parse("movielens_1m", &bytes, "::")?;
		if ratings.len() != 1_000_209 {
			return Err(Error::data_loss(format!(
				"movielens_1m decode produced {} rows, expected 1,000,209",
				ratings.len()
			)));
		}
		let mut features = Vec::with_capacity(ratings.len() * FEATURE_COUNT);
		let mut labels = Vec::with_capacity(ratings.len());
		for rating in &ratings {
			features.extend_from_slice(&[rating.user_id, rating.item_id, rating.timestamp]);
			labels.push(rating.rating);
		}
		Ok(Decoded {
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
		config: &Movielens1mConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Ok(Self {
			dataset: MatrixDataset::new(
				decoded.upload("features", engine)?,
				decoded.upload("ratings", engine)?,
			)?,
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
}

impl Dataset for Movielens1m {
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
