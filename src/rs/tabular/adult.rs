//! UCI Adult census-income classification as a device-resident
//! [`oa::data::Dataset`].
//!
//! `adult.zip` materializes through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `adult/1.0.0`;
//! `adult.data` (32,561 train rows) and `adult.test` (16,281 test
//! rows) decode to `[N, 92]` F32 features — six numeric fields plus
//! one-hot workclass, marital-status, occupation, relationship, race,
//! sex, and native-country (the `education` string column drops in
//! favor of the redundant `education-num`) — and `[N]` F32 income
//! labels `0..=1` (`<=50K`/`>50K`). Category tables are the sorted
//! union observed across both members, so encodings agree between
//! splits. `normalize` min-max scales each column to `[0, 1]`.

use std::collections::BTreeSet;

use oa::data::{Dataset, MatrixDataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::core::csv;
use crate::core::decode::{Decoded, MatrixData};
use crate::core::extract;
use crate::core::pack::{self, FetchProgress};
use crate::tabular::scale::normalize_columns;

#[path = "adult.gen.rs"]
mod generated;

pub(crate) use generated::ADULT_PACK;
pub use generated::{AdultConfig, AdultSplit, FEATURE_COUNT};

/// Column indices of the six numeric fields.
const NUMERIC: &[usize] = &[0, 2, 4, 10, 11, 12];
/// Column indices of the seven categorical fields (education dropped).
const CATEGORICAL: &[usize] = &[1, 5, 6, 7, 8, 9, 13];

fn member(split: AdultSplit) -> &'static str {
	match split {
		AdultSplit::Train => "adult.data",
		AdultSplit::Test => "adult.test",
	}
}

fn read_member(dir: &std::path::Path, member: &str) -> Result<Vec<Vec<String>>> {
	let mut body: Option<Vec<u8>> = None;
	extract::walk_zip_file(
		&dir.join("adult.zip"),
		|name| name == member,
		|name, data| {
			if name == member {
				body = Some(data.to_vec());
			}
			Ok(())
		},
	)?;
	let body = body.ok_or_else(|| Error::data_loss(format!("adult archive lacks {member}")))?;
	let records = csv::parse("adult", &body)?;
	Ok(
		records
			.into_iter()
			.filter(|record| record.len() == 15)
			.map(|record| {
				record
					.iter()
					.map(|field| field.trim().trim_end_matches('.').to_owned())
					.collect()
			})
			.collect(),
	)
}

/// Sorted-unique category tables for [`CATEGORICAL`] columns built
/// across `records` (both members feed one table set, so train and
/// test encode identically).
fn category_tables(records: &[Vec<String>]) -> Vec<Vec<String>> {
	let mut tables: Vec<BTreeSet<String>> = vec![BTreeSet::new(); CATEGORICAL.len()];
	for record in records {
		for (slot, &column) in CATEGORICAL.iter().enumerate() {
			tables[slot].insert(record[column].clone());
		}
	}
	tables
		.into_iter()
		.map(|set| set.into_iter().collect())
		.collect()
}

fn encode(
	pack: &str,
	records: &[Vec<String>],
	tables: &[Vec<String>],
) -> Result<(Vec<f32>, Vec<f32>)> {
	let mut features = Vec::with_capacity(records.len() * FEATURE_COUNT);
	let mut labels = Vec::with_capacity(records.len());
	for (line, record) in records.iter().enumerate() {
		for &column in NUMERIC {
			features.push(record[column].parse::<f32>().map_err(|_| {
				Error::data_loss(format!("{pack} row {line} field {column} is unparseable"))
			})?);
		}
		for (slot, &column) in CATEGORICAL.iter().enumerate() {
			let table = &tables[slot];
			let index = table
				.iter()
				.position(|value| *value == record[column])
				.ok_or_else(|| {
					Error::data_loss(format!(
						"{pack} row {line} field {column} holds an unknown category"
					))
				})?;
			for (hit, _) in table.iter().enumerate() {
				features.push(if hit == index { 1.0 } else { 0.0 });
			}
		}
		labels.push(match record[14].as_str() {
			"<=50K" => 0.0,
			">50K" => 1.0,
			other => {
				return Err(Error::data_loss(format!(
					"{pack} row {line} label '{other}' is unknown"
				)));
			}
		});
	}
	Ok((features, labels))
}

/// Device-resident Adult dataset implementing [`oa::data::Dataset`].
///
/// `features()` is `[N, 92]` F32 and `labels()` is `[N]` F32 income
/// indices `0..=1`.
pub struct Adult {
	dataset: MatrixDataset,
	split: AdultSplit,
}

impl Adult {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors, zip member errors, CSV
	/// decode errors, and an error when a field is unparseable.
	pub fn decode(config: &AdultConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus `Cancelled`
	/// when `progress` aborts a fetch.
	pub fn decode_with_progress(
		config: &AdultConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&ADULT_PACK, &root, progress)?;
		// Category tables span both members so train and test rows
		// encode against one vocabulary.
		let mut all = read_member(&dir, "adult.data")?;
		all.extend(read_member(&dir, "adult.test")?);
		let tables = category_tables(&all);
		let total: usize = NUMERIC.len() + tables.iter().map(Vec::len).sum::<usize>();
		if total != FEATURE_COUNT {
			return Err(Error::data_loss(format!(
				"adult one-hot width {total} disagrees with FEATURE_COUNT {FEATURE_COUNT}"
			)));
		}
		let records = read_member(&dir, member(config.split))?;
		let (mut features, labels) = encode("adult", &records, &tables)?;
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
		config: &AdultConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let decoded = Self::decode_with_progress(config, progress)?;
		Ok(Self {
			dataset: MatrixDataset::new(
				decoded.upload("features", engine)?,
				decoded.upload("labels", engine)?,
			)?,
			split: config.split,
		})
	}

	/// Full `[N, 92]` F32 feature matrix on the owning engine.
	#[must_use]
	pub const fn features(&self) -> &Matrix {
		self.dataset.x()
	}

	/// Full `[N]` F32 income label matrix on the owning engine.
	#[must_use]
	pub const fn labels(&self) -> &Matrix {
		self.dataset.y()
	}

	/// Selected split.
	pub const fn split(&self) -> AdultSplit {
		self.split
	}
}

impl Dataset for Adult {
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
