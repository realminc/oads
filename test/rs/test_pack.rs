//! Host-only pack resolution, fetch, and verification tests.

use std::fs;
use std::path::{Path, PathBuf};

use oads::{Pack, PackFile};

fn temp_dir(name: &str) -> PathBuf {
	let dir = std::env::temp_dir().join(format!("oa-pack-{name}-{}", std::process::id()));
	fs::create_dir_all(&dir).expect("create temp dir");
	dir
}

fn leak_str(value: String) -> &'static str {
	Box::leak(value.into_boxed_str())
}

fn shake_of(path: &Path) -> &'static str {
	leak_str(oads::hash_file(path).expect("hash payload").to_hex())
}

fn file_url(path: &Path) -> &'static str {
	leak_str(format!("file://{}", path.display()))
}

fn single_file_pack(
	name: &'static str,
	file_name: &'static str,
	payload: &[u8],
) -> (PathBuf, Pack) {
	let source = temp_dir(&format!("source-{name}"));
	let file = source.join(file_name);
	fs::write(&file, payload).expect("write payload");
	let pack = Pack {
		name,
		version: "0.0.1",
		license: "test",
		files: Box::leak(Box::new([PackFile {
			name: file_name,
			size: payload.len() as u64,
			shake256: shake_of(&file),
			urls: Box::leak(Box::new([file_url(&file)])),
		}])),
	};
	(source, pack)
}

#[test]
fn hash_file_matches_known_shake256() {
	let dir = temp_dir("hash");
	let path = dir.join("abc.bin");
	fs::write(&path, b"abc").expect("write");
	let digest = oads::hash_file(&path).expect("hash");
	assert_eq!(
		digest.to_hex(),
		"483366601360a8771c6863080cc4114d8db44530f8f1e1ee4f94ea37e78b5739"
	);
}

#[test]
fn ensure_pack_fetches_and_verifies_file_url() {
	let (_source, pack) = single_file_pack("fetchpack", "blob.bin", b"oa pack test payload");
	let root = temp_dir("cache");
	let dir = oads::ensure_pack(&pack, &root).expect("ensure");
	assert_eq!(dir, root.join("fetchpack").join("0.0.1"));
	assert_eq!(
		fs::read(dir.join("blob.bin")).expect("read member"),
		b"oa pack test payload"
	);
}

#[test]
fn ensure_pack_reuses_size_matching_member() {
	let (_source, pack) = single_file_pack("hitpack", "blob.bin", b"cached payload");
	let root = temp_dir("cache");
	let first = oads::ensure_pack(&pack, &root).expect("first ensure");
	// A size-matching member is a cache hit: it is not rehashed or refetched.
	let second = oads::ensure_pack(&pack, &root).expect("second ensure");
	assert_eq!(first, second);
}

#[test]
fn ensure_pack_rejects_digest_mismatch() {
	// The mirror serves different bytes than the manifest digest commits to.
	let source = temp_dir("source-badpack");
	let expected = source.join("expected.bin");
	let served = source.join("served.bin");
	fs::write(&expected, b"aaaaaaaaaaaaaaaa").expect("write");
	fs::write(&served, b"bbbbbbbbbbbbbbbb").expect("write");
	let pack = Pack {
		name: "badpack",
		version: "0.0.1",
		license: "test",
		files: Box::leak(Box::new([PackFile {
			name: "blob.bin",
			size: 16,
			shake256: shake_of(&expected),
			urls: Box::leak(Box::new([file_url(&served)])),
		}])),
	};
	let root = temp_dir("cache");
	let error = oads::ensure_pack(&pack, &root).expect_err("digest must be rejected");
	assert!(
		format!("{error}").contains("digest"),
		"unexpected error: {error}"
	);
}

#[test]
fn ensure_pack_reports_failed_fetch() {
	let dir = temp_dir("missing");
	let pack = Pack {
		name: "gonepack",
		version: "0.0.1",
		license: "test",
		files: Box::leak(Box::new([PackFile {
			name: "absent.bin",
			size: 4,
			shake256: "483366601360a8771c6863080cc4114d8db44530f8f1e1ee4f94ea37e78b5739",
			urls: Box::leak(Box::new([file_url(&dir.join("nonexistent.bin"))])),
		}])),
	};
	let root = temp_dir("cache");
	assert!(oads::ensure_pack(&pack, &root).is_err());
}

#[test]
fn ensure_pack_resumes_existing_partial() {
	// A stopped run left a valid prefix in the partial; the next run
	// continues from that offset instead of restarting.
	let payload: Vec<u8> = (0..(3 * 1024 * 1024 + 17))
		.map(|index| (index % 251) as u8)
		.collect();
	let (_source, pack) = single_file_pack("resumepack", "blob.bin", &payload);
	let root = temp_dir("cache");
	let dir = root.join("resumepack").join("0.0.1");
	fs::create_dir_all(&dir).expect("create pack dir");
	let partial = dir.join("blob.bin.partial");
	fs::write(&partial, &payload[..1024 * 1024]).expect("seed partial prefix");

	let mut samples = Vec::new();
	let out = oads::ensure_pack_with_progress(&pack, &root, &mut |sample| {
		samples.push((sample.file, sample.fetched, sample.total));
		true
	})
	.expect("resumed ensure");

	assert_eq!(
		fs::read(out.join("blob.bin")).expect("read member"),
		payload
	);
	assert!(!partial.exists(), "partial must be published");
	assert!(!samples.is_empty(), "progress samples expected");
	assert!(
		samples
			.iter()
			.all(|(file, _, total)| { *file == "blob.bin" && *total == payload.len() as u64 })
	);
	assert!(
		samples.windows(2).all(|pair| pair[0].1 < pair[1].1),
		"fetched bytes must increase: {samples:?}"
	);
	assert_eq!(samples.last().expect("last sample").1, payload.len() as u64);
}

#[test]
fn ensure_pack_promotes_complete_partial() {
	// A crash between fetch completion and publish left a whole member in
	// the partial; digest verification rescues it with no source access.
	let (_source, pack) = single_file_pack("donepack", "blob.bin", b"already fetched");
	let root = temp_dir("cache");
	let dir = root.join("donepack").join("0.0.1");
	fs::create_dir_all(&dir).expect("create pack dir");
	fs::write(dir.join("blob.bin.partial"), b"already fetched").expect("seed partial");
	let url = pack.files[0].urls[0];
	let source_path = url.strip_prefix("file://").expect("file url");
	fs::remove_file(source_path).expect("remove source");

	let out = oads::ensure_pack(&pack, &root).expect("rescue partial");
	assert_eq!(
		fs::read(out.join("blob.bin")).expect("read member"),
		b"already fetched"
	);
}

#[test]
fn ensure_pack_drops_oversized_partial() {
	// A partial larger than the manifest size cannot be a valid prefix;
	// it is discarded and the member is fetched fresh.
	let (_source, pack) = single_file_pack("overpack", "blob.bin", b"exact");
	let root = temp_dir("cache");
	let dir = root.join("overpack").join("0.0.1");
	fs::create_dir_all(&dir).expect("create pack dir");
	fs::write(dir.join("blob.bin.partial"), b"exact plus garbage tail")
		.expect("seed oversized partial");

	let out = oads::ensure_pack(&pack, &root).expect("refetch over oversized");
	assert_eq!(
		fs::read(out.join("blob.bin")).expect("read member"),
		b"exact"
	);
}

#[test]
fn ensure_pack_cancel_keeps_partial_for_resume() {
	// The callback aborts mid-copy; the partial stays on disk and a rerun
	// finishes the member from the recorded offset.
	let payload: Vec<u8> = (0..(4 * 1024 * 1024 + 3))
		.map(|index| (index % 253) as u8)
		.collect();
	let (_source, pack) = single_file_pack("cancelpack", "blob.bin", &payload);
	let root = temp_dir("cache");
	let dir = root.join("cancelpack").join("0.0.1");
	let partial = dir.join("blob.bin.partial");

	let mut calls = 0_usize;
	let error = oads::ensure_pack_with_progress(&pack, &root, &mut |_| {
		calls += 1;
		calls < 2
	})
	.expect_err("cancel must propagate");
	assert_eq!(error.kind(), oa::ErrorKind::Cancelled);
	assert!(!dir.join("blob.bin").exists(), "member must not publish");
	let prefix = fs::metadata(&partial).expect("partial kept").len() as usize;
	assert!(prefix > 0 && prefix < payload.len());

	let out = oads::ensure_pack(&pack, &root).expect("resume after cancel");
	assert_eq!(
		fs::read(out.join("blob.bin")).expect("read member"),
		payload
	);
	assert!(!partial.exists());
}

#[test]
fn terminal_progress_render_shows_bar_speed_and_eta() {
	use std::time::Duration;

	let sample = oads::FetchProgress {
		file: "blob.bin",
		fetched: 5 * 1024 * 1024,
		total: 10 * 1024 * 1024,
	};
	let line = oads::TerminalProgress::render(&sample, Duration::from_secs(2));
	assert!(line.contains("blob.bin"), "{line}");
	assert!(line.contains("50.0%"), "{line}");
	assert!(line.contains("5.0 MiB/10.0 MiB"), "{line}");
	assert!(line.contains("2.5 MiB/s"), "{line}");
	assert!(line.contains("elapsed 2s"), "{line}");
	assert!(line.contains("eta 2s"), "{line}");
}

#[test]
fn ensure_pack_cancel_skips_remaining_mirrors() {
	// Mirror failover survives ordinary fetch errors but must not swallow a
	// cancellation: after the callback aborts on the second mirror, the
	// third mirror is never attempted and nothing is published.
	let (_source, pack) = single_file_pack("mirrorspack", "blob.bin", b"payload");
	let valid = pack.files[0].urls[0];
	let missing = file_url(&temp_dir("missing-mirror").join("absent.bin"));
	let pack = Pack {
		name: "mirrorspack",
		version: "0.0.1",
		license: "test",
		files: Box::leak(Box::new([PackFile {
			urls: Box::leak(Box::new([missing, valid, valid])),
			..pack.files[0]
		}])),
	};
	let root = temp_dir("cache");
	let mut calls = 0_usize;
	let error = oads::ensure_pack_with_progress(&pack, &root, &mut |_| {
		calls += 1;
		calls > 1
	})
	.expect_err("cancel must propagate");
	assert_eq!(error.kind(), oa::ErrorKind::Cancelled);
	assert!(
		!root
			.join("mirrorspack")
			.join("0.0.1")
			.join("blob.bin")
			.exists(),
		"third mirror must not publish the member"
	);
}
