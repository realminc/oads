//! DeepMind mathematics dataset question/answer pairs.
//!
//! `mathematics_dataset-v1.0.tar.gz` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `math_dataset/1.0.0`. Each `<difficulty>/<module>__<task>.txt`
//! member carries alternating question/answer lines; Train merges the
//! `train-easy`, `train-medium`, and `train-hard` trees (the emitted
//! `difficulty` field records which), Validation reads `interpolate`,
//! and Test reads `extrapolate`. Labels index the question's module
//! (`algebra`, `arithmetic`, ...). `get_item` returns the
//! `[1, sum(K_i)]` end-to-end field row.

use oa::data::{Dataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::core::decode::Decoded;
use crate::core::extract;
use crate::core::pack::{self, FetchProgress};
use crate::text::recordl::{self, RecordData};

#[path = "math_dataset.gen.rs"]
mod generated;

pub(crate) use generated::MATH_DATASET_PACK;
pub use generated::{MathDatasetConfig, MathDatasetSplit};

const FIELDS: &[&str] = &["question", "answer", "difficulty"];
const MODULES: &[&str] = &[
	"algebra",
	"arithmetic",
	"calculus",
	"comparison",
	"measurement",
	"numbers",
	"polynomials",
	"probability",
];

fn directories(split: MathDatasetSplit) -> &'static [&'static str] {
	match split {
		MathDatasetSplit::Train => &["train-easy", "train-medium", "train-hard"],
		MathDatasetSplit::Validation => &["interpolate"],
		MathDatasetSplit::Test => &["extrapolate"],
	}
}

/// Parse alternating question/answer lines from a member stream into rows.
fn parse_rows(
	pack: &str,
	name: &str,
	member: &mut dyn std::io::Read,
	difficulty: &str,
	rows: &mut Vec<Vec<String>>,
	labels: &mut Vec<String>,
	count: Option<usize>,
) -> Result<()> {
	use std::io::BufRead;
	let module = name
		.rsplit('/')
		.next()
		.unwrap_or(name)
		.split("__")
		.next()
		.unwrap_or("");
	let label = MODULES
		.iter()
		.position(|known| *known == module)
		.ok_or_else(|| {
			Error::data_loss(format!(
				"{pack} member {name} carries unknown module '{module}'"
			))
		})?
		.to_string();
	let mut lines = std::io::BufReader::with_capacity(1 << 20, member)
		.lines()
		.map(|line| line.map_err(|error| Error::io("read math_dataset line", error)))
		.filter(|line| line.as_deref().map_or(true, |text| !text.is_empty()));
	while let Some(question) = lines.next() {
		let question = question?;
		let answer = lines.next().transpose()?.ok_or_else(|| {
			Error::data_loss(format!(
				"{pack} member {name} question lacks an answer line"
			))
		})?;
		rows.push(vec![question, answer, difficulty.to_owned()]);
		labels.push(label.clone());
		if count.is_some_and(|limit| rows.len() >= limit) {
			return Err(Error::cancelled(format!("{pack} row cap reached")));
		}
	}
	Ok(())
}

fn decode_member(
	pack: &str,
	tar_path: &std::path::Path,
	split: MathDatasetSplit,
	count: Option<usize>,
) -> Result<(Decoded, Vec<String>)> {
	let mut rows = Vec::new();
	let mut labels = Vec::new();
	let dirs = directories(split);
	extract::walk_tar_gz_reader(tar_path, |name, member| {
		let Some(dir) = name
			.strip_prefix("mathematics_dataset-v1.0/")
			.and_then(|rest| rest.split('/').next())
		else {
			return Ok(());
		};
		if name.ends_with(".txt") && dirs.contains(&dir) {
			let difficulty = dir.strip_prefix("train-").unwrap_or(dir);
			parse_rows(
				pack,
				name,
				member,
				difficulty,
				&mut rows,
				&mut labels,
				count,
			)?;
		}
		Ok(())
	})?;
	recordl::decode_rows(pack, FIELDS, &[], MODULES, true, &rows, &labels, "lines")
		.map(|(decoded, _)| (decoded, MODULES.iter().map(|m| (*m).to_owned()).collect()))
}

/// MathDataset question/answer dataset over 3 text field(s).
pub struct MathDataset {
	data: RecordData,
}

impl MathDataset {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and member decode errors.
	pub fn decode(config: &MathDatasetConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus
	/// `Cancelled` when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &MathDatasetConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&MATH_DATASET_PACK, &root, progress)?;
		let (decoded, _) = decode_member(
			"math_dataset",
			&dir.join("mathematics_dataset-v1.0.tar.gz"),
			config.split,
			config.count,
		)?;
		Ok(decoded)
	}

	/// [`load_on`](Self::load_on) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`load_on`](Self::load_on), plus
	/// `Cancelled` when the callback aborts a fetch.
	pub fn load_with_progress_on(
		engine: &Engine,
		config: &MathDatasetConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&MATH_DATASET_PACK, &root, progress)?;
		let (decoded, classes) = decode_member(
			"math_dataset",
			&dir.join("mathematics_dataset-v1.0.tar.gz"),
			config.split,
			config.count,
		)?;
		let data = RecordData::upload(engine, FIELDS, &decoded, classes, true)?;
		Ok(Self { data })
	}

	/// `[N, K]` F32 byte matrix for text field `field` (0-based, in
	/// the documented field order).
	///
	/// # Errors
	///
	/// Returns `invalid_argument` when `field` is out of range.
	pub fn field(&self, field: usize) -> Result<&Matrix> {
		self
			.data
			.fields
			.get(field)
			.ok_or_else(|| oa::Error::invalid_argument("math_dataset field index out of range"))
	}

	/// `[N]` F32 label matrix; `None` for unlabeled members.
	#[must_use]
	pub const fn labels(&self) -> Option<&Matrix> {
		self.data.labels.as_ref()
	}

	/// Class display names; label index `i` names `classes()[i]`.
	#[must_use]
	pub fn classes(&self) -> &[String] {
		&self.data.classes
	}
}

impl Dataset for MathDataset {
	fn len(&self) -> i64 {
		i64::try_from(self.data.rows()).unwrap_or(i64::MAX)
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		self.data.item(index)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		self.data.sample(index)
	}
}
