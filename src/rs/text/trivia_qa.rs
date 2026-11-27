//! TriviaQA reading-comprehension question/answer pairs (wikipedia domain).
//!
//! `triviaqa-rc.tar.gz` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `trivia_qa/1.0.0`. The selected `qa/wikipedia-*.json` member decodes
//! to one `[N, K_i]` F32 byte matrix per text field: `question`, the
//! canonical `Answer.Value` as `answer`, and the newline-joined
//! `Answer.Aliases` list as `aliases` (any alias is a valid eval
//! target). The upstream `test-without-answers` member carries no
//! `Answer` records, so `Test` emits `question` only. `get_item`
//! returns the `[1, sum(K_i)]` end-to-end field row.

use std::path::Path;

use oa::data::{Dataset, Sample};
use oa::matrix::Matrix;
use oa::{Error, Result, Runtime};

use crate::core::decode::Decoded;
use crate::core::extract;
use crate::core::pack::{self, FetchProgress};
use crate::text::recordl;

#[path = "trivia_qa.gen.rs"]
mod generated;

pub(crate) use generated::TRIVIA_QA_PACK;
pub use generated::{TriviaQaConfig, TriviaQaSplit};

const PACK: &str = "trivia_qa";
const ARCHIVE: &str = "triviaqa-rc.tar.gz";

fn member(split: TriviaQaSplit) -> &'static str {
	match split {
		TriviaQaSplit::Train => "qa/wikipedia-train.json",
		TriviaQaSplit::Validation => "qa/wikipedia-dev.json",
		TriviaQaSplit::Test => "qa/wikipedia-test-without-answers.json",
	}
}

fn fields(split: TriviaQaSplit) -> &'static [&'static str] {
	match split {
		TriviaQaSplit::Train | TriviaQaSplit::Validation => &["question", "answer", "aliases"],
		TriviaQaSplit::Test => &["question"],
	}
}

/// Extract the selected `qa/` member from the tar and flatten its
/// `Data` records into `(texts)` rows; `answer`/`aliases` fields are
/// empty for the test member.
fn decode_member(dir: &Path, split: TriviaQaSplit) -> Result<(Decoded, Vec<String>)> {
	let member = member(split);
	let mut body: Option<Vec<u8>> = None;
	extract::walk_tar_gz_reader(&dir.join(ARCHIVE), |name, stream| {
		if name.ends_with(member) {
			let mut data = Vec::new();
			stream
				.read_to_end(&mut data)
				.map_err(|error| Error::io("read trivia_qa member", error))?;
			body = Some(data);
		}
		Ok(())
	})?;
	let body =
		body.ok_or_else(|| Error::data_loss(format!("{PACK} archive lacks member {member}")))?;
	let document: serde_json::Value = serde_json::from_slice(&body).map_err(|error| {
		Error::data_loss(format!("{PACK} member {member} is not valid JSON: {error}"))
	})?;
	let data = document
		.get("Data")
		.and_then(|value| value.as_array())
		.ok_or_else(|| Error::data_loss(format!("{PACK} member {member} lacks Data records")))?;
	let labeled = !member.ends_with("without-answers.json");
	let mut rows: Vec<Vec<String>> = Vec::with_capacity(data.len());
	for (row, record) in data.iter().enumerate() {
		let question = record
			.get("Question")
			.and_then(|value| value.as_str())
			.ok_or_else(|| Error::data_loss(format!("{PACK} {member} row {row} lacks Question")))?;
		if !labeled {
			rows.push(vec![question.to_owned()]);
			continue;
		}
		let answer = record
			.get("Answer")
			.ok_or_else(|| Error::data_loss(format!("{PACK} {member} row {row} lacks Answer")))?;
		let value = answer
			.get("Value")
			.and_then(|value| value.as_str())
			.ok_or_else(|| Error::data_loss(format!("{PACK} {member} row {row} lacks Answer.Value")))?;
		let aliases = answer
			.get("Aliases")
			.and_then(|value| value.as_array())
			.map(|list| {
				list
					.iter()
					.filter_map(|value| value.as_str())
					.collect::<Vec<_>>()
					.join("\n")
			})
			.unwrap_or_default();
		rows.push(vec![question.to_owned(), value.to_owned(), aliases]);
	}
	recordl::decode_rows(PACK, fields(split), &[], &[], false, &rows, &[], member)
}

/// TriviaQA question/answer dataset over the wikipedia domain.
pub struct TriviaQa {
	data: recordl::RecordData,
}

impl TriviaQa {
	/// Fetch, verify, and decode the selected split on the host.
	///
	/// # Errors
	///
	/// Returns pack fetch/verification errors and member decode errors.
	pub fn decode(config: &TriviaQaConfig) -> Result<Decoded> {
		Self::decode_with_progress(config, &mut |_| true)
	}

	/// [`decode`](Self::decode) with a fetch progress callback.
	///
	/// # Errors
	///
	/// Returns the errors of [`decode`](Self::decode), plus
	/// `Cancelled` when `progress` aborts the fetch.
	pub fn decode_with_progress(
		config: &TriviaQaConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Decoded> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&TRIVIA_QA_PACK, &root, progress)?;
		let (decoded, _) = decode_member(&dir, config.split)?;
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
		config: &TriviaQaConfig,
		progress: &mut impl FnMut(FetchProgress) -> bool,
	) -> Result<Self> {
		let root = match &config.cache_dir {
			Some(dir) => dir.clone(),
			None => pack::data_root()?,
		};
		let dir = pack::ensure_pack_with_progress(&TRIVIA_QA_PACK, &root, progress)?;
		let (decoded, classes) = decode_member(&dir, config.split)?;
		let data = recordl::RecordData::upload(engine, fields(config.split), &decoded, classes, false)?;
		Ok(Self { data })
	}

	/// `[N, K]` F32 byte matrix for text field `field` (0-based, in
	/// the documented field order; `Test` carries `question` only).
	///
	/// # Errors
	///
	/// Returns `invalid_argument` when `field` is out of range.
	pub fn field(&self, field: usize) -> Result<&Matrix> {
		self
			.data
			.fields
			.get(field)
			.ok_or_else(|| Error::invalid_argument("trivia_qa field index out of range"))
	}

	/// `[N]` F32 label matrix; always `None` (answers are text, not
	/// class indices).
	#[must_use]
	pub const fn labels(&self) -> Option<&Matrix> {
		self.data.labels.as_ref()
	}
}

impl Dataset for TriviaQa {
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
