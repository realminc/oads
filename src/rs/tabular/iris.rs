//! Fisher's Iris flower data as a device-resident [`oa::data::Dataset`].
//!
//! `iris.zip` is materialized through [`ensure_pack`](crate::core::pack::ensure_pack)
//! under `iris/1.0.0`; the `iris.data` member decodes to a `[150, 4]` F32
//! feature matrix (sepal/petal length and width in centimetres) and a `[150]`
//! F32 label matrix of species indices `0..=2` into [`CLASSES`]. The pack
//! carries all 150 rows unsplit — partition with
//! [`oa::data::random_split`]. Per-sample slices collate to `[B, 4]` batches
//! through [`oa::data`].

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::core::csv;
use crate::core::decode::{Decoded, MatrixData};
use crate::core::extract;
use crate::core::pack::{self, FetchProgress};
use crate::tabular::scale::normalize_columns;

#[path = "iris.gen.rs"]
mod generated;

pub(crate) use generated::IRIS_PACK;
pub use generated::{CLASSES, FEATURE_COUNT, IrisConfig};

/// Device-resident Iris dataset implementing [`oa::data::Dataset`].
///
/// `features()` is `[150, 4]` F32 and `labels()` is `[150]` F32 species
/// indices `0..=2` into [`CLASSES`].
pub struct Iris {
	dataset: MatrixDataset,
}

impl Iris {
	/// Fetch, verify, and decode the pack on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, zip member errors, CSV decode
	/// errors, and an error when a row does not carry four numeric features
	/// and a known species label.
	pub fn decode(config: &IrisConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &IrisConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&IRIS_PACK, &root, progress)?;
		let zip =
			std::fs::read(dir.join("iris.zip")).map_err(|error| Error::io("read iris archive", error))?;

		let mut rows = Vec::new();
		extract::walk_zip(&zip, |name, data| {
			if name == "iris.data" {
				rows = csv::parse("iris", data)?;
			}
			Ok(())
		})?;
		if rows.is_empty() {
			return Err(Error::data_loss("iris pack is missing iris.data"));
		}

		let mut features = Vec::with_capacity(rows.len() * FEATURE_COUNT);
		let mut labels = Vec::with_capacity(rows.len());
		for record in &rows {
			if record.len() < 5 || record[4].is_empty() {
				continue;
			}
			for field in &record[..4] {
				features.push(
					field
						.parse::<f32>()
						.map_err(|_| Error::data_loss(format!("iris feature '{field}' is not numeric")))?,
				);
			}
			let species = record[4].strip_prefix("Iris-").unwrap_or(&record[4]);
			labels.push(class_index(species)?);
		}
		if labels.len() != 150 {
			return Err(Error::data_loss(format!(
				"iris decode produced {} rows, expected 150",
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
		config: &IrisConfig,
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

	/// Full `[150, 4]` F32 feature matrix on the owning engine.
	#[must_use]
	pub const fn features(&self) -> &Matrix {
		self.dataset.x()
	}

	/// Full `[150]` F32 label matrix on the owning engine; indices into
	/// [`CLASSES`].
	#[must_use]
	pub const fn labels(&self) -> &Matrix {
		self.dataset.y()
	}
}

impl Dataset for Iris {
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

fn class_index(species: &str) -> Result<f32> {
	CLASSES
		.iter()
		.position(|&name| name == species)
		.map(|index| index as f32)
		.ok_or_else(|| Error::data_loss(format!("iris species '{species}' is unknown")))
}
