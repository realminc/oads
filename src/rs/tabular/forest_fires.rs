//! Montesinho forest-fire burn area as a device-resident
//! [`oa::data::Dataset`].
//!
//! `forestfires.zip` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `forest_fires/1.0.0`; the `forestfires.csv` member decodes to a
//! `[517, 12]` F32 feature matrix — park cell `X`/`Y`, month and weekday
//! ordinals, the `FFMC`/`DMC`/`DC`/`ISI` fire indices, and
//! `temp`/`RH`/`wind`/`rain` — plus a `[517]` F32 burned-area regression
//! target in hectares. The pack carries all 517 rows unsplit — partition
//! with [`oa::data::random_split`]. Per-sample slices collate to
//! `[B, 12]` batches through [`oa::data`].

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::core::csv;
use crate::core::decode::{Decoded, MatrixData};
use crate::core::extract;
use crate::core::pack::{self, FetchProgress};
use crate::tabular::scale::normalize_columns;

#[path = "forest_fires.gen.rs"]
mod generated;

pub(crate) use generated::FOREST_FIRES_PACK;
pub use generated::{FEATURE_COUNT, ForestFiresConfig};

const MONTHS: &[&str] = &[
	"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
];
const DAYS: &[&str] = &["sun", "mon", "tue", "wed", "thu", "fri", "sat"];

/// Device-resident forest-fire dataset implementing [`oa::data::Dataset`].
///
/// `features()` is `[517, 12]` F32 and `labels()` is `[517]` F32 burned
/// hectares.
pub struct ForestFires {
	dataset: MatrixDataset,
}

impl ForestFires {
	/// Fetch, verify, and decode the pack on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, zip member errors, CSV decode
	/// errors, and an error when a numeric field or calendar name is
	/// unparseable.
	pub fn decode(config: &ForestFiresConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &ForestFiresConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&FOREST_FIRES_PACK, &root, progress)?;
		let mut body: Option<Vec<u8>> = None;
		extract::walk_zip_file(
			&dir.join("forestfires.zip"),
			|name| name == "forestfires.csv",
			|name, data| {
				if name == "forestfires.csv" {
					body = Some(data.to_vec());
				}
				Ok(())
			},
		)?;
		let body =
			body.ok_or_else(|| Error::data_loss("forest_fires archive lacks forestfires.csv"))?;
		let records = csv::parse("forest_fires", &body)?;
		let mut rows = records.iter();
		rows
			.next()
			.ok_or_else(|| Error::data_loss("forest_fires csv is empty"))?;
		let mut features = Vec::with_capacity(517 * FEATURE_COUNT);
		let mut labels = Vec::with_capacity(517);
		for (line, record) in rows.enumerate() {
			let numeric = |index: usize| -> Result<f32> {
				record
					.get(index)
					.and_then(|field| field.parse::<f32>().ok())
					.ok_or_else(|| {
						Error::data_loss(format!(
							"forest_fires row {line} field {index} is unparseable"
						))
					})
			};
			if record.len() != 13 {
				return Err(Error::data_loss(format!(
					"forest_fires row {line} carries {} fields, expected 13",
					record.len()
				)));
			}
			features.push(numeric(0)?);
			features.push(numeric(1)?);
			let month = record[2].as_str();
			features.push(
				MONTHS
					.iter()
					.position(|name| *name == month)
					.map(|index| (index + 1) as f32)
					.ok_or_else(|| {
						Error::data_loss(format!(
							"forest_fires row {line} month '{month}' is unknown"
						))
					})?,
			);
			let day = record[3].as_str();
			features.push(
				DAYS
					.iter()
					.position(|name| *name == day)
					.map(|index| (index + 1) as f32)
					.ok_or_else(|| {
						Error::data_loss(format!("forest_fires row {line} day '{day}' is unknown"))
					})?,
			);
			for index in 4..12 {
				features.push(numeric(index)?);
			}
			labels.push(numeric(12)?);
		}
		if labels.len() != 517 {
			return Err(Error::data_loss(format!(
				"forest_fires decode produced {} rows, expected 517",
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
		config: &ForestFiresConfig,
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

	/// Full `[517, 12]` F32 feature matrix on the owning engine.
	#[must_use]
	pub const fn features(&self) -> &Matrix {
		self.dataset.x()
	}

	/// Full `[517]` F32 burned-area target matrix on the owning engine.
	#[must_use]
	pub const fn labels(&self) -> &Matrix {
		self.dataset.y()
	}
}

impl Dataset for ForestFires {
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
