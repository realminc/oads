//! Semantic Textual Similarity regression task (score 0-5) as a GLUE TSV pack.
//!
//! `STS-B.zip` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `glue_stsb/1.0.0`
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

#[path = "glue_stsb.gen.rs"]
mod generated;

pub(crate) use generated::GLUE_STSB_PACK;
pub use generated::{GlueStsbConfig, GlueStsbSplit};

const SPEC: GlueSpec = GlueSpec {
	header: true,
	text_cols: &[7, 8],
	label_col: 9,
	label_names: &[],
	classes: &[],
};

fn member(split: GlueStsbSplit) -> &'static str {
	match split {
		GlueStsbSplit::Train => "train.tsv",
		GlueStsbSplit::Validation => "dev.tsv",
	}
}

/// STS-B: samples over 2 text fields + labels.
pub struct GlueStsb {
	data: GlueData,
}

impl GlueStsb {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and TSV decode errors.
	pub fn decode(config: &GlueStsbConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus
	/// `Cancelled` when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &GlueStsbConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => crate::core::pack::data_root()?,
		};
		let dir = crate::core::pack::ensure_pack_with_progress(&GLUE_STSB_PACK, &root, progress)?;
		let (decoded, _) = gluetsv::decode_member(
			"glue_stsb",
			&dir.join("STS-B.zip"),
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
		config: &GlueStsbConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let data = GlueData::load_on(
			engine,
			&GLUE_STSB_PACK,
			std::path::Path::new("STS-B.zip"),
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
	///
	/// STS-B labels are similarity scores in `[0, 5]`, not class
	/// indices.
	#[must_use]
	pub const fn labels(&self) -> &Matrix {
		&self.data.labels
	}
}

impl Dataset for GlueStsb {
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
