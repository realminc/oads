//! Versioned, content-addressed dataset packs.
//!
//! A [`Pack`] is an immutable manifest compiled into the owning dataset
//! module. [`ensure_pack`] materializes a pack under a cache root —
//! `$OA_DATA_DIR`, `~/.oads`, or `$XDG_CACHE_HOME/oa/data` — fetching
//! missing members through `curl` and verifying every fetched byte against the
//! manifest SHAKE256-256 digest before publishing it into the pack directory.
//! A cache hit checks the recorded byte size only; delete the pack directory
//! to force a refetch and full re-verification.
//!
//! Downloads write to `*.partial` members and resume from the recorded
//! byte length (`curl -C -`); a completed-but-unpublished partial is
//! verified in place instead of refetched. [`FetchProgress`] samples stream
//! to [`ensure_pack_with_progress`] callers for progress reporting, and a
//! callback returning `false` cancels the fetch while keeping the partial.

use std::fs;
use std::io::{IsTerminal, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use oa::crypto::{Hash, Hasher};
use oa::{Error, Result};

const HASH_CHUNK: usize = 1 << 20;
const FETCH_POLL: Duration = Duration::from_millis(100);

/// One fetch progress sample reported while materializing a pack member.
#[derive(Clone, Copy, Debug)]
pub struct FetchProgress {
	/// Member file name within the pack.
	pub file: &'static str,
	/// Bytes written to the partial so far, including a resumed prefix.
	pub fetched: u64,
	/// Expected final byte count from the manifest.
	pub total: u64,
}

/// Terminal renderer for [`FetchProgress`] samples.
///
/// Mirrors `oads._progress.FetchProgress` on the Python side: on a TTY it
/// rewrites one line per member with a bar, percentage, byte counts,
/// speed, elapsed, and ETA; on pipes it emits a start and a complete
/// line. Pass `|sample| renderer.report(sample)` as the
/// [`ensure_pack_with_progress`] or `load_with_progress` callback.
pub struct TerminalProgress {
	file: Option<&'static str>,
	start: Instant,
	last: Instant,
	tty: bool,
}

impl Default for TerminalProgress {
	fn default() -> Self {
		Self::new()
	}
}

impl TerminalProgress {
	const WIDTH: usize = 28;
	const THROTTLE: Duration = Duration::from_millis(100);

	/// Create a renderer writing to stderr.
	pub fn new() -> Self {
		Self {
			file: None,
			start: Instant::now(),
			last: Instant::now(),
			tty: std::io::stderr().is_terminal(),
		}
	}

	/// Format one sample as the single-line bar shown on a TTY.
	pub fn render(sample: &FetchProgress, elapsed: Duration) -> String {
		let fetched = sample.fetched as f64;
		let total = sample.total as f64;
		let seconds = elapsed.as_secs_f64();
		let speed = if seconds > 0.0 {
			fetched / seconds
		} else {
			0.0
		};
		let eta = if speed > 0.0 {
			(total - fetched) / speed
		} else {
			0.0
		};
		let ratio = if total > 0.0 {
			(fetched / total).min(1.0)
		} else {
			1.0
		};
		let done = (ratio * Self::WIDTH as f64) as usize;
		format!(
			"{} {:5.1}%|{}{}| {}/{} {}/s elapsed {} eta {}",
			sample.file,
			ratio * 100.0,
			"█".repeat(done),
			"░".repeat(Self::WIDTH - done),
			Self::size(sample.fetched as f64),
			Self::size(sample.total as f64),
			Self::size(speed),
			Self::duration(seconds),
			Self::duration(eta),
		)
	}

	/// Record one [`FetchProgress`] sample; always returns `true`.
	///
	/// Cancellation is an application concern — wire a signal handler to
	/// the callback when a Rust binary needs abort-to-partial behavior.
	pub fn report(&mut self, sample: FetchProgress) -> bool {
		let now = Instant::now();
		if self.file != Some(sample.file) {
			if self.tty && self.file.is_some() {
				eprintln!();
			}
			self.file = Some(sample.file);
			self.start = now;
			self.last = now;
			if !self.tty {
				eprintln!("oads: downloading {}", sample.file);
			}
		}
		if !self.tty {
			if sample.fetched >= sample.total {
				eprintln!("oads: {} complete", sample.file);
			}
			return true;
		}
		if now.duration_since(self.last) < Self::THROTTLE && sample.fetched < sample.total {
			return true;
		}
		self.last = now;
		eprint!(
			"\r{}   ",
			Self::render(&sample, now.duration_since(self.start))
		);
		if sample.fetched >= sample.total {
			eprintln!();
		}
		true
	}

	fn size(mut value: f64) -> String {
		for unit in ["B", "KiB", "MiB", "GiB"] {
			if value < 1024.0 || unit == "GiB" {
				return if unit == "B" {
					format!("{} {unit}", value as u64)
				} else {
					format!("{value:.1} {unit}")
				};
			}
			value /= 1024.0;
		}
		unreachable!()
	}

	fn duration(seconds: f64) -> String {
		let seconds = seconds.max(0.0) as u64;
		let minutes = seconds / 60;
		let hours = minutes / 60;
		if hours > 0 {
			format!("{hours}h{:02}m", minutes % 60)
		} else if minutes > 0 {
			format!("{minutes}m{:02}s", seconds % 60)
		} else {
			format!("{seconds}s")
		}
	}
}

/// One content-addressed member of a [`Pack`].
pub struct PackFile {
	/// File name inside the pack directory.
	pub name: &'static str,
	/// Exact byte length of the payload.
	pub size: u64,
	/// SHAKE256-256 digest of the payload as lowercase hexadecimal.
	pub shake256: &'static str,
	/// Mirror URLs tried in order until one verifies. `file://` is supported.
	pub urls: &'static [&'static str],
}

/// Immutable named-and-versioned file set compiled into a dataset module.
pub struct Pack {
	/// Dataset name; becomes the first cache path segment.
	pub name: &'static str,
	/// Pack version; becomes the second cache path segment.
	pub version: &'static str,
	/// License identifier for the upstream payload; texts live in `license/`.
	pub license: &'static str,
	/// Every member required for the pack to be complete.
	pub files: &'static [PackFile],
}

/// Resolve the dataset cache root.
///
/// Prefers `OA_DATA_DIR`, then `~/.oads`, then `$XDG_CACHE_HOME/oa/data`,
/// then `~/.cache/oa/data`, and creates the directory on first use.
///
/// # Errors
///
/// Returns an error when no cache location can be resolved or the directory
/// cannot be created.
pub fn data_root() -> Result<PathBuf> {
	let home = || {
		std::env::var_os("HOME")
			.filter(|dir| !dir.is_empty())
			.map(PathBuf::from)
	};
	let root = match std::env::var_os("OA_DATA_DIR") {
		Some(dir) if !dir.is_empty() => PathBuf::from(dir),
		_ => match home() {
			Some(home) => home.join(".oads"),
			None => match std::env::var_os("XDG_CACHE_HOME") {
				Some(dir) if !dir.is_empty() => PathBuf::from(dir).join("oa").join("data"),
				_ => {
					return Err(Error::not_found(
						"dataset cache root is unresolvable: set OA_DATA_DIR or HOME",
					));
				}
			},
		},
	};
	fs::create_dir_all(&root).map_err(|error| Error::io("create dataset cache root", error))?;
	Ok(root)
}

/// Materialize `pack` under `root`, fetching and verifying missing members.
///
/// Returns the pack directory `root/<name>/<version>`. A member already
/// present at the recorded byte size is accepted without rehashing; a missing
/// or size-mismatched member is downloaded to a temporary file, verified
/// against the manifest digest, and renamed into place.
///
/// # Errors
///
/// Returns an error when the directory cannot be created, every mirror fails,
/// or a fetched member's size or digest does not match the manifest.
pub fn ensure_pack(pack: &Pack, root: &Path) -> Result<PathBuf> {
	ensure_pack_with_progress(pack, root, &mut |_| true)
}

/// [`ensure_pack`] with a fetch progress callback.
///
/// The callback receives [`FetchProgress`] samples roughly every 100 ms
/// during downloads; returning `false` cancels the fetch with a
/// `Cancelled` error and keeps the partial for a later resume.
///
/// # Errors
///
/// Returns the same errors as [`ensure_pack`], plus `Cancelled` when the
/// callback aborts a fetch.
pub fn ensure_pack_with_progress(
	pack: &Pack,
	root: &Path,
	progress: &mut impl FnMut(FetchProgress) -> bool,
) -> Result<PathBuf> {
	let dir = root.join(pack.name).join(pack.version);
	fs::create_dir_all(&dir).map_err(|error| Error::io("create pack directory", error))?;
	for file in pack.files {
		ensure_file(&dir, file, progress)?;
	}
	Ok(dir)
}

/// Stream a file through SHAKE256-256.
///
/// # Errors
///
/// Returns an error when the file cannot be opened or read.
pub fn hash_file(path: &Path) -> Result<Hash> {
	let mut file = fs::File::open(path).map_err(|error| Error::io("open for hashing", error))?;
	let mut hasher = Hasher::new();
	let mut buffer = vec![0_u8; HASH_CHUNK];
	loop {
		let read = file
			.read(&mut buffer)
			.map_err(|error| Error::io("read for hashing", error))?;
		if read == 0 {
			break;
		}
		hasher.update(&buffer[..read])?;
	}
	Ok(hasher.finalize())
}

fn ensure_file(
	dir: &Path,
	file: &PackFile,
	progress: &mut impl FnMut(FetchProgress) -> bool,
) -> Result<()> {
	let target = dir.join(file.name);
	if let Ok(meta) = fs::metadata(&target) {
		if meta.len() == file.size {
			return Ok(());
		}
		fs::remove_file(&target).map_err(|error| Error::io("remove stale pack member", error))?;
	}
	let expected = Hash::new_from_hex(file.shake256).map_err(|_| {
		Error::internal(format!(
			"pack manifest digest for {} is not valid hex",
			file.name
		))
	})?;
	let partial = dir.join(format!("{}.partial", file.name));
	let mut last = None;
	for url in file.urls {
		match fetch_and_verify(url, file, &partial, &expected, progress) {
			Ok(()) => {
				fs::rename(&partial, &target).map_err(|error| Error::io("publish pack member", error))?;
				return Ok(());
			}
			Err(error) => {
				if error.kind() == oa::ErrorKind::Cancelled {
					return Err(error);
				}
				// Keep resumable partials; drop verified-corrupt or oversized
				// data so the next mirror or run starts clean.
				let length = fs::metadata(&partial).map(|meta| meta.len()).unwrap_or(0);
				if error.kind() == oa::ErrorKind::DataLoss || length > file.size {
					let _ = fs::remove_file(&partial);
				}
				last = Some(error);
			}
		}
	}
	Err(last.unwrap_or_else(|| {
		Error::unavailable(format!("pack member {} declares no mirrors", file.name))
	}))
}

fn fetch_and_verify(
	url: &str,
	file: &PackFile,
	partial: &Path,
	expected: &Hash,
	progress: &mut impl FnMut(FetchProgress) -> bool,
) -> Result<()> {
	fetch(url, file, partial, progress)?;
	let length = fs::metadata(partial)
		.map_err(|error| Error::io("inspect fetched pack member", error))?
		.len();
	if length != file.size {
		return Err(Error::data_loss(format!(
			"pack member {url} fetched {length} bytes, expected {}",
			file.size
		)));
	}
	let digest = hash_file(partial)?;
	if digest.as_bytes() != expected.as_bytes() {
		return Err(Error::data_loss(format!(
			"pack member {url} digest {} does not match manifest",
			digest.to_hex()
		)));
	}
	Ok(())
}

fn fetch(
	url: &str,
	file: &PackFile,
	partial: &Path,
	progress: &mut impl FnMut(FetchProgress) -> bool,
) -> Result<()> {
	if let Some(source) = url.strip_prefix("file://") {
		return copy_local(Path::new(source), file, partial, progress);
	}
	let existing = fs::metadata(partial).map(|meta| meta.len()).unwrap_or(0);
	if existing > file.size {
		fs::remove_file(partial).map_err(|error| Error::io("remove oversized partial", error))?;
	} else if existing == file.size && existing > 0 {
		// A previous run completed the download but crashed before publish;
		// let the digest check decide instead of refetching.
		return Ok(());
	}
	let mut child = Command::new("curl")
		.args(["-fLsS", "--retry", "2", "-C", "-", "-o"])
		.arg(partial)
		.arg(url)
		.stderr(Stdio::piped())
		.spawn()
		.map_err(|error| Error::io("spawn curl", error))?;
	loop {
		match child.try_wait() {
			Ok(Some(status)) => {
				if !status.success() {
					let stderr = child
						.stderr
						.take()
						.and_then(|mut pipe| {
							let mut text = String::new();
							pipe.read_to_string(&mut text).ok().map(|_| text)
						})
						.unwrap_or_default();
					return Err(Error::unavailable(format!(
						"fetch {url} failed: {}",
						stderr.trim()
					)));
				}
				let fetched = fs::metadata(partial).map(|meta| meta.len()).unwrap_or(0);
				let _ = progress(FetchProgress {
					file: file.name,
					fetched,
					total: file.size,
				});
				return Ok(());
			}
			Ok(None) => {
				let fetched = fs::metadata(partial).map(|meta| meta.len()).unwrap_or(0);
				if !progress(FetchProgress {
					file: file.name,
					fetched,
					total: file.size,
				}) {
					let _ = child.kill();
					let _ = child.wait();
					return Err(Error::cancelled(format!("fetch {} cancelled", file.name)));
				}
				std::thread::sleep(FETCH_POLL);
			}
			Err(error) => {
				let _ = child.kill();
				let _ = child.wait();
				return Err(Error::io("wait on curl", error));
			}
		}
	}
}

fn copy_local(
	source: &Path,
	file: &PackFile,
	partial: &Path,
	progress: &mut impl FnMut(FetchProgress) -> bool,
) -> Result<()> {
	let mut fetched = fs::metadata(partial).map(|meta| meta.len()).unwrap_or(0);
	if fetched > file.size {
		fs::remove_file(partial).map_err(|error| Error::io("remove oversized partial", error))?;
		fetched = 0;
	}
	if fetched == file.size && fetched > 0 {
		return Ok(());
	}
	let mut input =
		fs::File::open(source).map_err(|error| Error::io("open local pack member source", error))?;
	input
		.seek(SeekFrom::Start(fetched))
		.map_err(|error| Error::io("seek local pack member source", error))?;
	let mut output = fs::OpenOptions::new()
		.create(true)
		.append(true)
		.open(partial)
		.map_err(|error| Error::io("open pack partial", error))?;
	let mut buffer = vec![0_u8; HASH_CHUNK];
	loop {
		let read = input
			.read(&mut buffer)
			.map_err(|error| Error::io("read local pack member source", error))?;
		if read == 0 {
			return Ok(());
		}
		output
			.write_all(&buffer[..read])
			.map_err(|error| Error::io("write pack partial", error))?;
		fetched += read as u64;
		if !progress(FetchProgress {
			file: file.name,
			fetched,
			total: file.size,
		}) {
			return Err(Error::cancelled(format!("fetch {} cancelled", file.name)));
		}
	}
}
