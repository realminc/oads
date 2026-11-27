//! Statlog German credit risk as a device-resident [`oa::data::Dataset`].
//!
//! `germancredit.zip` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `german_credit/1.0.0`; the `german.data-numeric` member decodes to a
//! `[1000, 24]` F32 feature matrix (the UCI numeric recoding of the 20
//! attribute set plus indicator dummies) and a `[1000]` F32 risk-label
//! matrix — class `1` maps to index `0` (good) and class `2` to index `1`
//! (bad). The pack carries all 1000 rows unsplit — partition with
//! [`oa::data::random_split`]. Per-sample slices collate to `[B, 24]`
//! batches through [`oa::data`].

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Error, Result, Runtime};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::extract;
use crate::core::pack::{self, FetchProgress};
use crate::tabular::scale::normalize_columns;

#[path = "german_credit.gen.rs"]
mod generated;

pub(crate) use generated::GERMAN_CREDIT_PACK;
pub use generated::{CLASSES, FEATURE_COUNT, GermanCreditConfig};

/// Device-resident German credit dataset implementing
/// [`oa::data::Dataset`].
///
/// `features()` is `[1000, 24]` F32 and `labels()` is `[1000]` F32 risk
/// indices `0..=1` into [`CLASSES`].
pub struct GermanCredit {
	dataset: MatrixDataset,
}

impl GermanCredit {
	/// Fetch, verify, and decode the pack on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, zip member errors, and an
	/// error when a numeric field or class label is unparseable.
	pub fn decode(config: &GermanCreditConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled` when
	/// `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &GermanCreditConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&GERMAN_CREDIT_PACK, &root, progress)?;
		let mut body: Option<Vec<u8>> = None;
		extract::walk_zip_file(
			&dir.join("germancredit.zip"),
			|name| name == "german.data-numeric",
			|name, data| {
				if name == "german.data-numeric" {
					body = Some(data.to_vec());
				}
				Ok(())
			},
		)?;
		let body =
			body.ok_or_else(|| Error::data_loss("german_credit archive lacks german.data-numeric"))?;
		let text = std::str::from_utf8(&body)
			.map_err(|_| Error::data_loss("german_credit member is not valid UTF-8"))?;
		let mut features = Vec::with_capacity(1000 * FEATURE_COUNT);
		let mut labels = Vec::with_capacity(1000);
		for (line_no, line) in text.lines().enumerate() {
			let fields: Vec<&str> = line.split_whitespace().collect();
			if fields.is_empty() {
				continue;
			}
			if fields.len() != FEATURE_COUNT + 1 {
				return Err(Error::data_loss(format!(
					"german_credit row {line_no} carries {} fields, expected {}",
					fields.len(),
					FEATURE_COUNT + 1
				)));
			}
			for field in &fields[..FEATURE_COUNT] {
				features.push(field.parse::<f32>().map_err(|_| {
					Error::data_loss(format!("german_credit row {line_no} field is unparseable"))
				})?);
			}
			labels.push(match fields[FEATURE_COUNT] {
				"1" => 0.0,
				"2" => 1.0,
				other => {
					return Err(Error::data_loss(format!(
						"german_credit row {line_no} class '{other}' is unknown"
					)));
				}
			});
		}
		if labels.len() != 1000 {
			return Err(Error::data_loss(format!(
				"german_credit decode produced {} rows, expected 1000",
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
		engine: &Runtime,
		config: &GermanCreditConfig,
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

	/// Full `[1000, 24]` F32 feature matrix on the owning engine.
	#[must_use]
	pub const fn features(&self) -> &Matrix {
		self.dataset.x()
	}

	/// Full `[1000]` F32 risk-label matrix on the owning engine.
	#[must_use]
	pub const fn labels(&self) -> &Matrix {
		self.dataset.y()
	}
}

impl Dataset for GermanCredit {
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
