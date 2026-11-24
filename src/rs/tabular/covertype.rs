//! UCI Covertype forest cover-type classification as a device-resident
//! [`oa::data::Dataset`].
//!
//! `covertype.zip` materializes through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `covertype/1.0.0`;
//! the gzipped `covtype.data` member decodes to `[N, 54]` F32 features
//! (10 numeric measures plus 4 wilderness-area and 40 soil-type binary
//! indicators) and `[N]` F32 cover-type indices `0..=6` (raw labels
//! `1..=7` minus one). `normalize` min-max scales each column to
//! `[0, 1]`.

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::core::csv;
use crate::core::decode::{Decoded, MatrixData};
use crate::core::extract;
use crate::core::pack::{self, FetchProgress};
use crate::tabular::scale::normalize_columns;

#[path = "covertype.gen.rs"]
mod generated;

pub(crate) use generated::COVERTYPE_PACK;
pub use generated::{CovertypeConfig, FEATURE_COUNT};

/// Device-resident Covertype dataset implementing [`oa::data::Dataset`].
///
/// `features()` is `[581_012, 54]` F32 and `labels()` is `[581_012]`
/// F32 cover-type indices `0..=6`.
pub struct Covertype {
	dataset: MatrixDataset,
}

impl Covertype {
	/// Fetch, verify, and decode the pack on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, zip member errors,
	/// gzip decode errors, and an error when a field is unparseable.
	pub fn decode(config: &CovertypeConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled`
	/// when `progress` aborts a fetch.
	pub fn decode_with_progress(
		config: &CovertypeConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&COVERTYPE_PACK, &root, progress)?;
		let mut body: Option<Vec<u8>> = None;
		extract::walk_zip_file(
			&dir.join("covertype.zip"),
			|name| name == "covtype.data.gz",
			|name, data| {
				if name == "covtype.data.gz" {
					body = Some(data.to_vec());
				}
				Ok(())
			},
		)?;
		let body = body.ok_or_else(|| Error::data_loss("covertype archive lacks covtype.data.gz"))?;
		let records = csv::parse("covertype", &extract::gunzip_bytes(&body)?)?;
		let mut features = Vec::with_capacity(records.len() * FEATURE_COUNT);
		let mut labels = Vec::with_capacity(records.len());
		for (line, record) in records.iter().enumerate() {
			if record.len() != FEATURE_COUNT + 1 {
				return Err(Error::data_loss(format!(
					"covertype row {line} carries {} fields, expected {}",
					record.len(),
					FEATURE_COUNT + 1
				)));
			}
			let numeric = |index: usize| -> Result<f32> {
				record[index].parse::<f32>().map_err(|_| {
					Error::data_loss(format!("covertype row {line} field {index} is unparseable"))
				})
			};
			for index in 0..FEATURE_COUNT {
				features.push(numeric(index)?);
			}
			labels.push(numeric(FEATURE_COUNT)? - 1.0);
		}
		if config.normalize {
			normalize_columns(&mut features, FEATURE_COUNT);
		}
		Ok(Decoded {
			matrices: vec![
				MatrixData {
					name: "features",
					shape: vec![labels.len(), FEATURE_COUNT],
					values: features,
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
		config: &CovertypeConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Ok(Self {
			dataset: MatrixDataset::new(
				decoded.upload("features", engine)?,
				decoded.upload("labels", engine)?,
			)?,
		})
	}

	/// Full `[581_012, 54]` F32 feature matrix on the owning engine.
	#[must_use]
	pub const fn features(&self) -> &Matrix {
		self.dataset.x()
	}

	/// Full `[581_012]` F32 cover-type label matrix on the owning engine.
	#[must_use]
	pub const fn labels(&self) -> &Matrix {
		self.dataset.y()
	}
}

impl Dataset for Covertype {
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
