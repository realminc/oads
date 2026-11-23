//! Stanford Sentiment Treebank binary sentiment task as a GLUE TSV pack.
//!
//! `SST-2.zip` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `glue_sst2/1.0.0`
//! and the selected split member decodes through
//! `gluetsv`: `text1` padded F32 byte
//! matrices plus the `[N]` F32 label matrix. `get_item` returns the
//! `[F, K]` stacked text rows for one sample.

use oa::data::{Dataset, Sample};
use oa::matrix::Matrix;
use oa::{Engine, Result};

use crate::core::decode::Decoded;
use crate::core::pack::FetchProgress;
use crate::text::gluetsv::{self, GlueData, GlueSpec};

#[path = "glue_sst2.gen.rs"]
mod generated;

pub(crate) use generated::GLUE_SST2_PACK;
pub use generated::{GlueSst2Config, GlueSst2Split};

const SPEC: GlueSpec = GlueSpec {
	header: true,
	text_cols: &[0],
	label_col: 1,
	label_names: &[],
	classes: &["negative", "positive"],
};

fn member(split: GlueSst2Split) -> &'static str {
	match split {
		GlueSst2Split::Train => "train.tsv",
		GlueSst2Split::Validation => "dev.tsv",
	}
}

/// SST-2: samples over 1 text field + labels.
pub struct GlueSst2 {
	data: GlueData,
}

impl GlueSst2 {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and TSV decode errors.
	pub fn decode(config: &GlueSst2Config) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus
	/// `Cancelled` when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &GlueSst2Config,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => crate::core::pack::data_root()?,
		};
		let dir = crate::core::pack::ensure_pack_with_progress(&GLUE_SST2_PACK, &root, progress)?;
		let (decoded, _) = gluetsv::decode_member(
			"glue_sst2",
			&dir.join("SST-2.zip"),
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
		engine: &Engine,
		config: &GlueSst2Config,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let data = GlueData::load_on(
			engine,
			&GLUE_SST2_PACK,
			std::path::Path::new("SST-2.zip"),
			member(config.split),
			&SPEC,
			config.cache_dir.clone(),
			progress,
		)?;
		Ok(Self { data })
	}

	/// `[N, K]` F32 first-text-field byte matrix.
	#[must_use]
	pub const fn text1(&self) -> &Matrix {
		&self.data.text1
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

impl Dataset for GlueSst2 {
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
