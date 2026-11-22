//! Pack-member decompression shared by loaders.
//!
//! [`gunzip`] decompresses a gzip member to bytes. [`tar_member`] locates one
//! regular file inside an uncompressed tar archive without copying.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use flate2::read::GzDecoder;

use oa::{Error, Result};

const TAR_BLOCK: usize = 512;

/// Decompress a gzip pack member to bytes.
pub(crate) fn gunzip(path: &Path) -> Result<Vec<u8>> {
	let file = File::open(path).map_err(|error| Error::io("open pack member", error))?;
	let mut bytes = Vec::new();
	GzDecoder::new(file)
		.read_to_end(&mut bytes)
		.map_err(|error| Error::io("decompress pack member", error))?;
	Ok(bytes)
}

/// Borrow the contents of regular file `name` inside an uncompressed tar
/// archive. `name` matches either the full member path or its last component,
/// so archive directory prefixes do not leak into the manifest contract.
///
/// # Errors
///
/// Returns `data_loss` for truncated or malformed headers and `not_found`
/// when no regular member matches `name`.
pub(crate) fn tar_member<'a>(tar: &'a [u8], name: &str) -> Result<&'a [u8]> {
	let mut offset = 0;
	while offset + TAR_BLOCK <= tar.len() {
		let header = &tar[offset..offset + TAR_BLOCK];
		if header.iter().all(|&b| b == 0) {
			break;
		}
		let size = parse_octal(&header[124..136])
			.ok_or_else(|| Error::data_loss("tar member size is not a valid octal field"))?;
		offset += TAR_BLOCK;
		let end = offset + size;
		if end > tar.len() {
			return Err(Error::data_loss("tar member extends past the archive"));
		}
		let member_name = member_name(header)?;
		let typeflag = header[156];
		if (typeflag == b'0' || typeflag == 0)
			&& (member_name == name || member_name.ends_with(&format!("/{name}")))
		{
			return Ok(&tar[offset..end]);
		}
		offset = end.div_ceil(TAR_BLOCK) * TAR_BLOCK;
	}
	Err(Error::not_found(format!(
		"tar archive holds no regular member {name}"
	)))
}

fn member_name(header: &[u8]) -> Result<&str> {
	let raw = &header[..100];
	let len = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
	std::str::from_utf8(&raw[..len])
		.map_err(|_| Error::data_loss("tar member name is not valid UTF-8"))
}

fn parse_octal(field: &[u8]) -> Option<usize> {
	let digits = field
		.iter()
		.copied()
		.filter(|b| b.is_ascii_digit())
		.collect::<Vec<_>>();
	let mut value = 0_usize;
	for digit in digits {
		if digit > b'7' {
			return None;
		}
		value = value
			.checked_mul(8)?
			.checked_add(usize::from(digit - b'0'))?;
	}
	Some(value)
}
