//! Host-side zip walker contract: stored and deflated members, directory
//! skipping, central-directory order, and malformed-archive rejection.

use std::io::Write;

use super::*;

/// Build a minimal zip archive: one local header + central entry per
/// (name, payload, method) triple. `method` 8 deflates the payload.
fn make_zip(entries: &[(&str, &[u8], u16)]) -> Vec<u8> {
	let mut zip = Vec::new();
	let mut central = Vec::new();
	for &(name, payload, method) in entries {
		let data: Vec<u8> = if method == 8 {
			let mut encoder = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::fast());
			encoder.write_all(payload).expect("deflate");
			encoder.finish().expect("deflate finish")
		} else {
			payload.to_vec()
		};
		let local_offset = zip.len() as u32;
		zip.extend_from_slice(&LOCAL_SIGNATURE.to_le_bytes());
		zip.extend_from_slice(&[20, 0, 0, 0]); // version, flags
		zip.extend_from_slice(&method.to_le_bytes());
		zip.extend_from_slice(&[0; 4]); // time, date
		zip.extend_from_slice(&[0; 4]); // crc (unused by the walker)
		zip.extend_from_slice(&(data.len() as u32).to_le_bytes());
		zip.extend_from_slice(&(payload.len() as u32).to_le_bytes());
		zip.extend_from_slice(&(name.len() as u16).to_le_bytes());
		zip.extend_from_slice(&[0; 2]); // extra len
		zip.extend_from_slice(name.as_bytes());
		zip.extend_from_slice(&data);

		central.extend_from_slice(&CENTRAL_SIGNATURE.to_le_bytes());
		central.extend_from_slice(&[20, 0, 20, 0, 0, 0]); // versions, flags
		central.extend_from_slice(&method.to_le_bytes());
		central.extend_from_slice(&[0; 4]); // time, date
		central.extend_from_slice(&[0; 4]); // crc
		central.extend_from_slice(&(data.len() as u32).to_le_bytes());
		central.extend_from_slice(&(payload.len() as u32).to_le_bytes());
		central.extend_from_slice(&(name.len() as u16).to_le_bytes());
		central.extend_from_slice(&[0; 8]); // extra, comment, disk, internal attrs
		central.extend_from_slice(&[0; 4]); // external attrs
		central.extend_from_slice(&local_offset.to_le_bytes());
		central.extend_from_slice(name.as_bytes());
	}
	let cd_offset = zip.len() as u32;
	let cd_len = central.len() as u32;
	zip.extend_from_slice(&central);
	zip.extend_from_slice(&EOCD_SIGNATURE.to_le_bytes());
	zip.extend_from_slice(&[0; 4]); // disk numbers
	zip.extend_from_slice(&(entries.len() as u16).to_le_bytes());
	zip.extend_from_slice(&(entries.len() as u16).to_le_bytes());
	zip.extend_from_slice(&cd_len.to_le_bytes());
	zip.extend_from_slice(&cd_offset.to_le_bytes());
	zip.extend_from_slice(&[0; 2]); // comment len
	zip
}

fn collect(zip: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
	let mut seen = Vec::new();
	walk_zip(zip, |name, data| {
		seen.push((name.to_string(), data.to_vec()));
		Ok(())
	})?;
	Ok(seen)
}

#[test]
fn zip_walks_stored_and_deflated_members_in_order() {
	let zip = make_zip(&[
		("a/one.txt", b"first".as_slice(), 0),
		("a/two.txt", b"second-deflated".as_slice(), 8),
		("b/three.txt", b"third".as_slice(), 0),
	]);
	let seen = collect(&zip).expect("walk");
	assert_eq!(
		seen,
		vec![
			("a/one.txt".to_string(), b"first".to_vec()),
			("a/two.txt".to_string(), b"second-deflated".to_vec()),
			("b/three.txt".to_string(), b"third".to_vec()),
		]
	);
}

#[test]
fn zip_skips_directory_entries() {
	let zip = make_zip(&[
		("dir/", b"".as_slice(), 0),
		("dir/clip.wav", b"payload".as_slice(), 8),
	]);
	let seen = collect(&zip).expect("walk");
	assert_eq!(seen.len(), 1);
	assert_eq!(seen[0].0, "dir/clip.wav");
}

#[test]
fn zip_rejects_truncated_and_foreign_archives() {
	assert!(collect(b"not a zip").is_err());
	let zip = make_zip(&[("a.txt", b"x".as_slice(), 0)]);
	assert!(collect(&zip[..zip.len() - 4]).is_err());
	// Unsupported compression method surfaces as data_loss, not a panic.
	let zip = make_zip(&[("a.txt", b"x".as_slice(), 99)]);
	assert!(collect(&zip).is_err());
}
