//! Multi-Genre Natural Language Inference task as a GLUE TSV pack.
//!
//! `MNLI.zip` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `glue_mnli/1.0.0`
//! and the selected split member decodes through
//! `gluetsv`: ``text1` and `text2`` padded F32 byte
//! matrices plus the `[N]` F32 label matrix. `get_item` returns the
//! `[F, K]` stacked text rows for one sample.

use oa::data::{Dataset, Sample};
use oa::matrix::Matrix;
use oa::{Result, Runtime};

use crate::core::decode::Decoded;
use crate::core::pack::FetchProgress;
use crate::text::gluetsv::{self, GlueData, GlueSpec};

#[path = "glue_mnli.gen.rs"]
mod generated;

pub(crate) use generated::GLUE_MNLI_PACK;
pub use generated::{GlueMnliConfig, GlueMnliSplit};

const SPEC: GlueSpec = GlueSpec {
	header: true,
	text_cols: &[8, 9],
	label_col: 11,
	label_names: &["entailment", "neutral", "contradiction"],
	classes: &["entailment", "neutral", "contradiction"],
};

fn member(split: GlueMnliSplit) -> &'static str {
	match split {
		GlueMnliSplit::Train => "train.tsv",
		GlueMnliSplit::Validation => "dev_matched.tsv",
		GlueMnliSplit::Test => "dev_mismatched.tsv",
	}
}

/// MNLI: samples over 2 text fields + labels.
pub struct GlueMnli {
	data: GlueData,
}

impl GlueMnli {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and TSV decode errors.
	pub fn decode(config: &GlueMnliConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus
	/// `Cancelled` when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &GlueMnliConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => crate::core::pack::data_root()?,
		};
		let dir = crate::core::pack::ensure_pack_with_progress(&GLUE_MNLI_PACK, &root, progress)?;
		let (decoded, _) = gluetsv::decode_member(
			"glue_mnli",
			&dir.join("MNLI.zip"),
			member(config.split),
			&SPEC,
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
		engine: &Runtime,
		config: &GlueMnliConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let data = GlueData::load_on(
			engine,
			&GLUE_MNLI_PACK,
			std::path::Path::new("MNLI.zip"),
			member(config.split),
			&SPEC,
			config.cache_dir.clone(),
			progress,
		)?;
		Ok(Self { data })
	}

	/// `[N, K]` F32 first-text-field byte matrix.
	///
	/// Pair tasks stack `text1`/`text2` in `get_item`.
	#[must_use]
	pub const fn text1(&self) -> &Matrix {
		&self.data.text1
	}

	/// `[N, K]` F32 second-text-field byte matrix; `None` on
	/// single-field tasks.
	#[must_use]
	pub const fn text2(&self) -> Option<&Matrix> {
		self.data.text2.as_ref()
	}

	/// `[N]` F32 label or score matrix.
	#[must_use]
	pub const fn labels(&self) -> &Matrix {
		&self.data.labels
	}

	/// Class display names; label index `i` names `classes()[i]`.
	#[must_use]
	pub fn classes(&self) -> &[String] {
		&self.data.classes
	}
}

impl Dataset for GlueMnli {
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
