//! Titanic passenger survival table as a device-resident
//! [`oa::data::Dataset`].
//!
//! `titanic.csv` is materialized through [`ensure_pack`](crate::pack::ensure_pack)
//! under `titanic/1.0.0` and decodes to a `[891, 9]` F32 feature matrix —
//! pclass, sex, age, sibsp, parch, fare, and one-hot embarked — plus a `[891]`
//! F32 survival-label matrix. Missing age fills with the column median and
//! missing embarked fills S. The pack carries all 891 rows unsplit —
//! partition with [`oa::data::random_split`]. Per-sample slices collate to
//! `[B, 9]` batches through [`oa::data`].

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::csv;
use crate::decode::{Decoded, MatrixData};
use crate::pack::{self, FetchProgress};
use crate::tabular::scale::normalize_columns;

#[path = "titanic.gen.rs"]
mod generated;

pub(crate) use generated::TITANIC_PACK;
pub use generated::{FEATURE_COUNT, TitanicConfig};

/// Device-resident Titanic dataset implementing [`oa::data::Dataset`].
///
/// `features()` is `[891, 9]` F32 and `labels()` is `[891]` F32 survival
/// flags `0.0`/`1.0`.
pub struct Titanic {
	dataset: MatrixDataset,
}

impl Titanic {
	/// Fetch, verify, and decode the pack on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, CSV decode errors, and an
	/// error when the header misses a required column or a numeric field is
	/// unparseable.
	pub fn decode(config: &TitanicConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &TitanicConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&TITANIC_PACK, &root, progress)?;
		let csv_bytes = std::fs::read(dir.join("titanic.csv"))
			.map_err(|error| Error::io("read titanic csv", error))?;
		let records = csv::parse("titanic", &csv_bytes)?;
		let mut rows = records.iter();
		let header = rows
			.next()
			.ok_or_else(|| Error::data_loss("titanic csv is empty"))?;
		let col = |name: &str| -> Result<usize> {
			header
				.iter()
				.position(|field| field == name)
				.ok_or_else(|| Error::data_loss(format!("titanic csv has no {name} column")))
		};
		let (survived, pclass, sex, age, sibsp, parch, fare, embarked) = (
			col("Survived")?,
			col("Pclass")?,
			col("Sex")?,
			col("Age")?,
			col("SibSp")?,
			col("Parch")?,
			col("Fare")?,
			col("Embarked")?,
		);

		// Median age over the present values fills the missing cells; the two
		// empty embarked cells fill with the modal S.
		let ages: Vec<f32> = rows
			.clone()
			.filter_map(|record| record.get(age)?.parse::<f32>().ok())
			.collect();
		let mut sorted = ages;
		sorted.sort_by(f32::total_cmp);
		let median_age = *sorted
			.get(sorted.len() / 2)
			.ok_or_else(|| Error::data_loss("titanic csv has no numeric ages"))?;

		let mut features = Vec::with_capacity(891 * FEATURE_COUNT);
		let mut labels = Vec::with_capacity(891);
		for record in rows {
			let numeric = |index: usize| -> Result<f32> {
				record
					.get(index)
					.and_then(|field| field.parse::<f32>().ok())
					.ok_or_else(|| Error::data_loss("titanic numeric field is unparseable"))
			};
			labels.push(numeric(survived)?);
			features.push(numeric(pclass)?);
			features.push(match record.get(sex).map(String::as_str) {
				Some("female") => 1.0,
				Some("male") => 0.0,
				_ => return Err(Error::data_loss("titanic sex field is unknown")),
			});
			features.push(
				record
					.get(age)
					.and_then(|field| field.parse::<f32>().ok())
					.unwrap_or(median_age),
			);
			features.push(numeric(sibsp)?);
			features.push(numeric(parch)?);
			features.push(numeric(fare)?);
			let port = record.get(embarked).map(String::as_str).unwrap_or("");
			let port = if port.is_empty() { "S" } else { port };
			for flag in ["C", "Q", "S"] {
				features.push(if port == flag { 1.0 } else { 0.0 });
			}
		}
		if labels.len() != 891 {
			return Err(Error::data_loss(format!(
				"titanic decode produced {} rows, expected 891",
				labels.len()
			)));
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
		config: &TitanicConfig,
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

	/// Full `[891, 9]` F32 feature matrix on the owning engine.
	#[must_use]
	pub const fn features(&self) -> &Matrix {
		self.dataset.x()
	}

	/// Full `[891]` F32 survival-label matrix on the owning engine.
	#[must_use]
	pub const fn labels(&self) -> &Matrix {
		self.dataset.y()
	}
}

impl Dataset for Titanic {
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
