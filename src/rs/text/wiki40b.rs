//! Wiki-40B cleaned Wikipedia article corpus (English config).
//!
//! 140 TFRecord shards under `wiki40b/1.0.0` carry `tf.train.Example`
//! records with `text` and `wikidata_id` byte features. Article text keeps
//! the upstream `_START_ARTICLE_`/`_START_PARAGRAPH_`/`_NEWLINE_` section
//! markers. Articles flatten to a `[B]` F32 byte blob with `[N]`
//! `article_lengths` and `wikidata_id_lengths` tables (offsets exceed F32
//! integer precision past 16 MiB, so callers derive spans by cumulative
//! sum). `get_item` returns the `[1, len_i]` article row.

use std::path::Path;

use oa::data::{Dataset, Sample};
use oa::matrix::{self, Matrix};
use oa::{Error, Result, Runtime};

use crate::core::decode::{Decoded, MatrixData};
use crate::core::pack::{self, FetchProgress};
use crate::core::tfrecord;

#[path = "wiki40b.gen.rs"]
mod generated;

pub(crate) use generated::WIKI40B_PACK;
pub use generated::{Wiki40BConfig, Wiki40BSplit};

fn prefix(split: Wiki40BSplit) -> &'static str {
	match split {
		Wiki40BSplit::Train => "wiki40b-train.tfrecord-",
		Wiki40BSplit::Validation => "wiki40b-validation.tfrecord-",
		Wiki40BSplit::Test => "wiki40b-test.tfrecord-",
	}
}

/// Flatten every record under `prefix` into `text`/`wikidata_id` byte
/// blobs plus per-article length tables and a host prefix-sum index.
fn decode_members(
	pack: &str,
	dir: &Path,
	prefix: &str,
	count: Option<usize>,
) -> Result<(Decoded, Vec<u64>)> {
	let mut shards: Vec<_> = std::fs::read_dir(dir)
		.map_err(|error| Error::io("read pack directory", error))?
		.filter_map(|entry| entry.ok())
		.map(|entry| entry.file_name().to_string_lossy().into_owned())
		.filter(|name| name.starts_with(prefix))
		.collect();
	shards.sort();
	if shards.is_empty() {
		return Err(Error::data_loss(format!(
			"{pack} pack lacks {prefix} shards"
		)));
	}
	let mut text_blob: Vec<u8> = Vec::new();
	let mut id_blob: Vec<u8> = Vec::new();
	let mut text_lengths: Vec<f32> = Vec::new();
	let mut id_lengths: Vec<f32> = Vec::new();
	let mut offsets: Vec<u64> = Vec::new();
	'shards: for shard in shards {
		let mut capped = false;
		tfrecord::walk_tfrecord(&dir.join(&shard), |payload| {
			if count.is_some_and(|limit| text_lengths.len() >= limit) {
				capped = true;
				return Err(Error::cancelled(format!("{pack} row cap reached")));
			}
			let text = tfrecord::example_bytes(payload, "text").unwrap_or(&[]);
			let id = tfrecord::example_bytes(payload, "wikidata_id").unwrap_or(&[]);
			offsets.push(text_blob.len() as u64);
			text_blob.extend_from_slice(text);
			text_lengths.push(text.len() as f32);
			id_blob.extend_from_slice(id);
			id_lengths.push(id.len() as f32);
			Ok(())
		})?;
		if capped {
			break 'shards;
		}
	}
	if offsets.is_empty() {
		return Err(Error::data_loss(format!("{pack} shards carry no articles")));
	}
	let articles = text_lengths.len();
	Ok((
		Decoded {
			matrices: vec![
				MatrixData {
					name: "text",
					shape: vec![text_blob.len()],
					values: text_blob.iter().map(|&byte| f32::from(byte)).collect(),
				},
				MatrixData {
					name: "article_lengths",
					shape: vec![articles],
					values: text_lengths,
				},
				MatrixData {
					name: "wikidata_id",
					shape: vec![id_blob.len()],
					values: id_blob.iter().map(|&byte| f32::from(byte)).collect(),
				},
				MatrixData {
					name: "wikidata_id_lengths",
					shape: vec![articles],
					values: id_lengths,
				},
			],
			text: None,
			seq_len: 0,
		},
		offsets,
	))
}

/// Wiki-40B article stream: flat `[B]` byte blobs plus `[N]` lengths.
pub struct Wiki40B {
	text: Matrix,
	article_lengths: Matrix,
	wikidata_id: Matrix,
	wikidata_id_lengths: Matrix,
	offsets: Vec<u64>,
	rows: usize,
}

impl Wiki40B {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and shard decode errors.
	pub fn decode(config: &Wiki40BConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus
	/// `Cancelled` when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &Wiki40BConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&WIKI40B_PACK, &root, progress)?;
		let (decoded, _) = decode_members("wiki40b", &dir, prefix(config.split), config.count)?;
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
		config: &Wiki40BConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&WIKI40B_PACK, &root, progress)?;
		let (decoded, offsets) = decode_members("wiki40b", &dir, prefix(config.split), config.count)?;
		Ok(Self {
			rows: offsets.len(),
			text: decoded.upload("text", engine)?,
			article_lengths: decoded.upload("article_lengths", engine)?,
			wikidata_id: decoded.upload("wikidata_id", engine)?,
			wikidata_id_lengths: decoded.upload("wikidata_id_lengths", engine)?,
			offsets,
		})
	}

	/// `[B]` F32 flat article byte blob on the owning engine.
	#[must_use]
	pub const fn text(&self) -> &Matrix {
		&self.text
	}

	/// `[N]` F32 per-article byte lengths; cumulative sums give blob
	/// offsets.
	#[must_use]
	pub const fn article_lengths(&self) -> &Matrix {
		&self.article_lengths
	}

	/// `[B]` F32 flat Wikidata id byte blob on the owning engine.
	#[must_use]
	pub const fn wikidata_id(&self) -> &Matrix {
		&self.wikidata_id
	}

	/// `[N]` F32 per-article Wikidata id byte lengths.
	#[must_use]
	pub const fn wikidata_id_lengths(&self) -> &Matrix {
		&self.wikidata_id_lengths
	}
}

impl Dataset for Wiki40B {
	fn len(&self) -> i64 {
		i64::try_from(self.rows).unwrap_or(i64::MAX)
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		let start = *self
			.offsets
			.get(index as usize)
			.ok_or_else(|| Error::invalid_argument("wiki40b index out of range"))?;
		let end = self
			.offsets
			.get(index as usize + 1)
			.copied()
			.unwrap_or(self.text.shape()[0] as u64);
		matrix::slice(
			&self.text,
			0,
			i64::try_from(start).unwrap_or(i64::MAX),
			i64::try_from(end).unwrap_or(i64::MAX),
		)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		Ok(Sample::new(self.get_item(index)?))
	}
}
