//! Pack-member decompression shared by loaders.
//!
//! [`gunzip`] decompresses a gzip member to bytes. [`tar_member`] locates one
//! regular file inside an uncompressed tar archive without copying.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use bzip2::read::BzDecoder;
use flate2::read::{DeflateDecoder, GzDecoder};

use oa::{Error, Result};

const TAR_BLOCK: usize = 512;
const EOCD_SIGNATURE: u32 = 0x0605_4b50;
const CENTRAL_SIGNATURE: u32 = 0x0201_4b50;
const LOCAL_SIGNATURE: u32 = 0x0403_4b50;
const EOCD_MIN_LEN: usize = 22;
const EOCD_SCAN: usize = 66_000;
const EOCD64_SIGNATURE: u32 = 0x0606_4b50;
const EOCD64_LOCATOR_SIGNATURE: u32 = 0x0706_4b50;
const EOCD64_LOCATOR_LEN: u64 = 20;
const EOCD64_FIXED_LEN: usize = 56;
const ZIP64_EXTRA_ID: u16 = 0x0001;
const CENTRAL_FIXED_LEN: usize = 46;
const LOCAL_FIXED_LEN: usize = 30;

/// Decompress a gzip pack member to bytes.
pub(crate) fn gunzip(path: &Path) -> Result<Vec<u8>> {
	let file = File::open(path).map_err(|error| Error::io("open pack member", error))?;
	let mut bytes = Vec::new();
	GzDecoder::new(file)
		.read_to_end(&mut bytes)
		.map_err(|error| Error::io("decompress pack member", error))?;
	Ok(bytes)
}

/// Decompress a gzip byte stream, such as a member extracted from a zip.
pub(crate) fn gunzip_bytes(bytes: &[u8]) -> Result<Vec<u8>> {
	let mut out = Vec::new();
	GzDecoder::new(bytes)
		.read_to_end(&mut out)
		.map_err(|error| Error::io("decompress gzip member", error))?;
	Ok(out)
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

/// Invoke `visit` once per regular-file member of an uncompressed tar
/// archive, in archive order. `visit` receives the member path and its bytes.
///
/// # Errors
///
/// Returns `data_loss` for truncated or malformed headers, plus any error
/// `visit` returns.
pub(crate) fn walk_tar(tar: &[u8], mut visit: impl FnMut(&str, &[u8]) -> Result<()>) -> Result<()> {
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
		if header[156] == b'0' || header[156] == 0 {
			visit(member_name(header)?, &tar[offset..end])?;
		}
		offset = end.div_ceil(TAR_BLOCK) * TAR_BLOCK;
	}
	Ok(())
}

/// Invoke `visit` once per regular-file member of a zip archive, in central
/// directory order. `visit` receives the member path and its decompressed
/// bytes; directory entries are skipped. Stored (`0`) and deflated (`8`)
/// members are supported.
///
/// # Errors
///
/// Returns `data_loss` for truncated archives, missing directory records,
/// unsupported compression methods, or decompressed sizes that disagree
/// with the central directory.
pub(crate) fn walk_zip(zip: &[u8], mut visit: impl FnMut(&str, &[u8]) -> Result<()>) -> Result<()> {
	if zip.len() < EOCD_MIN_LEN {
		return Err(Error::data_loss("zip archive is truncated"));
	}
	let scan_start = zip.len().saturating_sub(EOCD_SCAN);
	let eocd = zip[scan_start..]
		.windows(EOCD_MIN_LEN)
		.rposition(|window| le32(&window[..4]) == EOCD_SIGNATURE)
		.map(|index| scan_start + index)
		.ok_or_else(|| Error::data_loss("zip archive has no end of central directory"))?;
	let entry_count = le16(&zip[eocd + 10..eocd + 12]) as usize;
	let mut offset = le32(&zip[eocd + 16..eocd + 20]) as usize;
	for _ in 0..entry_count {
		if offset + CENTRAL_FIXED_LEN > zip.len() || le32(&zip[offset..offset + 4]) != CENTRAL_SIGNATURE
		{
			return Err(Error::data_loss("zip central directory is truncated"));
		}
		let method = le16(&zip[offset + 10..offset + 12]);
		let compressed = le32(&zip[offset + 20..offset + 24]) as usize;
		let expected = le32(&zip[offset + 24..offset + 28]) as usize;
		let name_len = le16(&zip[offset + 28..offset + 30]) as usize;
		let extra_len = le16(&zip[offset + 30..offset + 32]) as usize;
		let comment_len = le16(&zip[offset + 32..offset + 34]) as usize;
		let local = le32(&zip[offset + 42..offset + 46]) as usize;
		let name_end = offset + CENTRAL_FIXED_LEN + name_len;
		if name_end > zip.len() {
			return Err(Error::data_loss("zip member name extends past the archive"));
		}
		let name = std::str::from_utf8(&zip[offset + CENTRAL_FIXED_LEN..name_end])
			.map_err(|_| Error::data_loss("zip member name is not valid UTF-8"))?;
		offset = name_end
			.checked_add(extra_len)
			.and_then(|end| end.checked_add(comment_len))
			.ok_or_else(|| Error::data_loss("zip central directory overflows"))?;
		if name.ends_with('/') {
			continue;
		}
		if local + LOCAL_FIXED_LEN > zip.len() || le32(&zip[local..local + 4]) != LOCAL_SIGNATURE {
			return Err(Error::data_loss("zip local header is truncated"));
		}
		let local_name = le16(&zip[local + 26..local + 28]) as usize;
		let local_extra = le16(&zip[local + 28..local + 30]) as usize;
		let data_start = local + LOCAL_FIXED_LEN + local_name + local_extra;
		let data_end = data_start
			.checked_add(compressed)
			.ok_or_else(|| Error::data_loss("zip member size overflows usize"))?;
		if data_end > zip.len() {
			return Err(Error::data_loss("zip member extends past the archive"));
		}
		let data = &zip[data_start..data_end];
		match method {
			0 => {
				if data.len() != expected {
					return Err(Error::data_loss("zip stored member size disagrees"));
				}
				visit(name, data)?;
			}
			8 => {
				let mut bytes = Vec::with_capacity(expected);
				DeflateDecoder::new(data)
					.read_to_end(&mut bytes)
					.map_err(|error| Error::io("decompress zip member", error))?;
				if bytes.len() != expected {
					return Err(Error::data_loss("zip inflated member size disagrees"));
				}
				visit(name, &bytes)?;
			}
			other => {
				return Err(Error::data_loss(format!(
					"zip member uses unsupported compression method {other}"
				)));
			}
		}
	}
	Ok(())
}

/// Invoke `visit` once per regular-file member of a gzip-compressed tar
/// archive, in archive order. The decoder streams members one at a time, so
/// archives larger than host memory still decode; `visit` receives the
/// member path and its bytes.
///
/// # Errors
///
/// Returns I/O errors opening or reading `path`, `data_loss` for truncated
/// or malformed headers, plus any error `visit` returns.
pub(crate) fn walk_tar_gz(
	path: &Path,
	mut visit: impl FnMut(&str, &[u8]) -> Result<()>,
) -> Result<()> {
	let file = File::open(path).map_err(|error| Error::io("open pack member", error))?;
	walk_tar_stream(GzDecoder::new(file), &mut visit)
}

/// Invoke `visit` once per regular-file member of a bzip2-compressed tar
/// archive, in archive order. Streaming equivalent of [`walk_tar_gz`] for
/// `.tar.bz2` packs.
///
/// # Errors
///
/// Returns I/O errors opening or reading `path`, `data_loss` for truncated
/// or malformed headers, plus any error `visit` returns.
pub(crate) fn walk_tar_bz2(
	path: &Path,
	mut visit: impl FnMut(&str, &[u8]) -> Result<()>,
) -> Result<()> {
	let file = File::open(path).map_err(|error| Error::io("open pack member", error))?;
	walk_tar_stream(BzDecoder::new(file), &mut visit)
}

fn walk_tar_stream(
	mut reader: impl Read,
	visit: &mut impl FnMut(&str, &[u8]) -> Result<()>,
) -> Result<()> {
	let mut header = [0u8; TAR_BLOCK];
	let mut read_exact = |buf: &mut [u8]| {
		reader
			.read_exact(buf)
			.map_err(|error| Error::io("read tar member", error))
	};
	loop {
		match read_exact(&mut header) {
			Ok(()) => {}
			Err(_) => break,
		}
		if header.iter().all(|&b| b == 0) {
			break;
		}
		let size = parse_octal(&header[124..136])
			.ok_or_else(|| Error::data_loss("tar member size is not a valid octal field"))?;
		let typeflag = header[156];
		let name = member_name(&header)?.to_string();
		if typeflag == b'0' || typeflag == 0 {
			let mut data = vec![0u8; size];
			read_exact(&mut data)?;
			let padding = size.div_ceil(TAR_BLOCK) * TAR_BLOCK - size;
			if padding > 0 {
				let mut skip = vec![0u8; padding];
				read_exact(&mut skip)?;
			}
			visit(&name, &data)?;
		} else {
			// Skip non-file members: read-and-discard in bounded chunks.
			let mut remaining = size.div_ceil(TAR_BLOCK) * TAR_BLOCK;
			let mut chunk = [0u8; 64 * 1024];
			while remaining > 0 {
				let take = remaining.min(chunk.len());
				read_exact(&mut chunk[..take])?;
				remaining -= take;
			}
		}
	}
	Ok(())
}

/// Invoke `visit` once per regular-file member of a zip archive at `path`
/// whose name satisfies `include`, in central directory order. The central
/// directory is read once, then each selected member is located and
/// inflated on demand, so archives larger than host memory still decode.
/// `visit` receives the member path and its decompressed bytes.
///
/// # Errors
///
/// Returns I/O errors reading `path`, `data_loss` for truncated archives,
/// missing directory records, unsupported compression methods, or
/// decompressed sizes that disagree with the central directory, plus any
/// error `visit` returns.
pub(crate) fn walk_zip_file(
	path: &Path,
	mut include: impl FnMut(&str) -> bool,
	mut visit: impl FnMut(&str, &[u8]) -> Result<()>,
) -> Result<()> {
	use std::io::Seek;

	let mut file = File::open(path).map_err(|error| Error::io("open zip archive", error))?;
	let entries = zip_entries(&mut file)?
		.into_iter()
		.filter(|entry| !entry.name.ends_with('/') && include(&entry.name))
		.collect::<Vec<_>>();

	for entry in entries {
		let (data_start, _) = local_data_range(&mut file, &entry)?;
		file
			.seek(std::io::SeekFrom::Start(data_start))
			.map_err(|error| Error::io("seek zip member data", error))?;
		let mut data = vec![
			0u8;
			usize::try_from(entry.compressed)
				.map_err(|_| Error::data_loss("zip member is too large to buffer"))?
		];
		file
			.read_exact(&mut data)
			.map_err(|error| Error::io("read zip member data", error))?;
		match entry.method {
			0 => {
				if data.len() as u64 != entry.expected {
					return Err(Error::data_loss("zip stored member size disagrees"));
				}
				visit(&entry.name, &data)?;
			}
			8 => {
				let mut bytes = Vec::with_capacity(entry.expected as usize);
				DeflateDecoder::new(&data[..])
					.read_to_end(&mut bytes)
					.map_err(|error| Error::io("decompress zip member", error))?;
				if bytes.len() as u64 != entry.expected {
					return Err(Error::data_loss("zip inflated member size disagrees"));
				}
				visit(&entry.name, &bytes)?;
			}
			other => {
				return Err(Error::data_loss(format!(
					"zip member uses unsupported compression method {other}"
				)));
			}
		}
	}
	Ok(())
}

/// Stream-decompress the zip member `name` at `path` into `out`, without
/// materializing the member in memory. For oversized members such as
/// VCTK's nested corpus archive where [`walk_zip_file`]'s per-member byte
/// vector is not affordable.
///
/// # Errors
///
/// Returns I/O errors reading `path` or writing `out`, `data_loss` for
/// malformed records, `not_found` when no member matches `name`,
/// unsupported compression methods, or a written size that disagrees with
/// the central directory.
pub(crate) fn extract_zip_member_to_file(path: &Path, name: &str, out: &Path) -> Result<()> {
	use std::io::{Seek, SeekFrom, Write};

	let mut file = File::open(path).map_err(|error| Error::io("open zip archive", error))?;
	let entry = zip_entries(&mut file)?
		.into_iter()
		.find(|entry| entry.name == name)
		.ok_or_else(|| Error::not_found(format!("zip member '{name}' is absent")))?;
	let (data_start, _) = local_data_range(&mut file, &entry)?;
	file
		.seek(SeekFrom::Start(data_start))
		.map_err(|error| Error::io("seek zip member data", error))?;
	let mut writer = std::io::BufWriter::new(
		File::create(out).map_err(|error| Error::io("create extracted member", error))?,
	);
	let mut limited = std::io::Read::take(&mut file, entry.compressed);
	let written = match entry.method {
		0 => std::io::copy(&mut limited, &mut writer)
			.map_err(|error| Error::io("copy stored zip member", error))?,
		8 => {
			let mut decoder = DeflateDecoder::new(&mut limited);
			std::io::copy(&mut decoder, &mut writer)
				.map_err(|error| Error::io("decompress zip member", error))?
		}
		other => {
			return Err(Error::data_loss(format!(
				"zip member uses unsupported compression method {other}"
			)));
		}
	};
	writer
		.flush()
		.map_err(|error| Error::io("finish extracted member", error))?;
	if written != entry.expected {
		return Err(Error::data_loss("zip member size disagrees"));
	}
	Ok(())
}

/// One central-directory record of a zip archive.
struct ZipEntry {
	name: String,
	method: u16,
	/// Local-header offset, compressed size, and uncompressed size. `u64`
	/// because ZIP64 archives exceed the 32-bit fields.
	local: u64,
	compressed: u64,
	expected: u64,
}

/// Parse the central directory of the zip in `file` (zip32 and zip64),
/// returning one [`ZipEntry`] per member.
fn zip_entries(file: &mut File) -> Result<Vec<ZipEntry>> {
	use std::io::{Seek, SeekFrom};

	let file_len = file
		.metadata()
		.map_err(|error| Error::io("stat zip archive", error))?
		.len();
	if file_len < EOCD_MIN_LEN as u64 {
		return Err(Error::data_loss("zip archive is truncated"));
	}
	let scan = (file_len as usize).min(EOCD_SCAN);
	file
		.seek(SeekFrom::Start(file_len - scan as u64))
		.map_err(|error| Error::io("seek zip directory", error))?;
	let mut tail = vec![0u8; scan];
	file
		.read_exact(&mut tail)
		.map_err(|error| Error::io("read zip directory", error))?;
	let eocd = tail
		.windows(EOCD_MIN_LEN)
		.rposition(|window| le32(&window[..4]) == EOCD_SIGNATURE)
		.ok_or_else(|| Error::data_loss("zip archive has no end of central directory"))?;
	let mut entry_count = u64::from(le16(&tail[eocd + 10..eocd + 12]));
	let mut dir_offset = u64::from(le32(&tail[eocd + 16..eocd + 20]));

	// ZIP64: sentinel 32-bit fields defer to the ZIP64 end of central
	// directory, located through the 20-byte locator before the EOCD.
	if entry_count == 0xFFFF || dir_offset == 0xFFFF_FFFF {
		let loc_off = file_len - (scan - eocd) as u64 - EOCD64_LOCATOR_LEN;
		let mut locator = [0u8; EOCD64_LOCATOR_LEN as usize];
		file
			.seek(SeekFrom::Start(loc_off))
			.map_err(|error| Error::io("seek zip64 locator", error))?;
		file
			.read_exact(&mut locator)
			.map_err(|error| Error::io("read zip64 locator", error))?;
		if le32(&locator[..4]) != EOCD64_LOCATOR_SIGNATURE {
			return Err(Error::data_loss("zip64 archive has no locator"));
		}
		let eocd64_off = le64(&locator[8..16])?;
		let mut eocd64 = [0u8; EOCD64_FIXED_LEN];
		file
			.seek(SeekFrom::Start(eocd64_off))
			.map_err(|error| Error::io("seek zip64 directory", error))?;
		file
			.read_exact(&mut eocd64)
			.map_err(|error| Error::io("read zip64 directory", error))?;
		if le32(&eocd64[..4]) != EOCD64_SIGNATURE {
			return Err(Error::data_loss("zip64 end of directory is truncated"));
		}
		entry_count = le64(&eocd64[32..40])?;
		dir_offset = le64(&eocd64[48..56])?;
	}

	file
		.seek(SeekFrom::Start(dir_offset))
		.map_err(|error| Error::io("seek zip members", error))?;
	let mut entries = Vec::with_capacity(entry_count as usize);
	for _ in 0..entry_count {
		let mut central = [0u8; CENTRAL_FIXED_LEN];
		file
			.read_exact(&mut central)
			.map_err(|error| Error::io("read zip directory entry", error))?;
		if le32(&central[..4]) != CENTRAL_SIGNATURE {
			return Err(Error::data_loss("zip central directory is truncated"));
		}
		let method = le16(&central[10..12]);
		let mut compressed = u64::from(le32(&central[20..24]));
		let mut expected = u64::from(le32(&central[24..28]));
		let name_len = le16(&central[28..30]) as usize;
		let extra_len = le16(&central[30..32]) as usize;
		let comment_len = le16(&central[32..34]) as usize;
		let mut local = u64::from(le32(&central[42..46]));
		let mut name = vec![0u8; name_len];
		file
			.read_exact(&mut name)
			.map_err(|error| Error::io("read zip member name", error))?;
		let mut extra = vec![0u8; extra_len];
		file
			.read_exact(&mut extra)
			.map_err(|error| Error::io("read zip member extra", error))?;
		file
			.seek(SeekFrom::Current(comment_len as i64))
			.map_err(|error| Error::io("seek zip directory entry", error))?;
		let name = String::from_utf8(name)
			.map_err(|_| Error::data_loss("zip member name is not valid UTF-8"))?;

		// ZIP64 extended information: u64 values replace sentinel fields
		// in order (uncompressed, compressed, local offset, disk start).
		if expected == 0xFFFF_FFFF || compressed == 0xFFFF_FFFF || local == 0xFFFF_FFFF {
			let mut cursor = 0_usize;
			while cursor + 4 <= extra.len() {
				let id = le16(&extra[cursor..cursor + 2]);
				let len = le16(&extra[cursor + 2..cursor + 4]) as usize;
				let data = &extra[cursor + 4..];
				if id == ZIP64_EXTRA_ID {
					let mut slot = 0_usize;
					if expected == 0xFFFF_FFFF {
						expected = le64(&data[slot..slot + 8])?;
						slot += 8;
					}
					if compressed == 0xFFFF_FFFF {
						compressed = le64(&data[slot..slot + 8])?;
						slot += 8;
					}
					if local == 0xFFFF_FFFF {
						local = le64(&data[slot..slot + 8])?;
					}
					break;
				}
				cursor += 4 + len;
			}
		}
		entries.push(ZipEntry {
			name,
			method,
			local,
			compressed,
			expected,
		});
	}
	Ok(entries)
}

/// Byte range of `entry`'s compressed data within `file`, resolved through
/// the member's local header (whose name/extra lengths differ from the
/// central copy).
fn local_data_range(file: &mut File, entry: &ZipEntry) -> Result<(u64, u64)> {
	use std::io::{Seek, SeekFrom};

	let mut header = [0u8; LOCAL_FIXED_LEN];
	file
		.seek(SeekFrom::Start(entry.local))
		.map_err(|error| Error::io("seek zip member", error))?;
	file
		.read_exact(&mut header)
		.map_err(|error| Error::io("read zip member header", error))?;
	if le32(&header[..4]) != LOCAL_SIGNATURE {
		return Err(Error::data_loss("zip local header is truncated"));
	}
	let local_name = u64::from(le16(&header[26..28]));
	let local_extra = u64::from(le16(&header[28..30]));
	let data_start = entry.local + LOCAL_FIXED_LEN as u64 + local_name + local_extra;
	Ok((data_start, entry.compressed))
}

fn member_name(header: &[u8]) -> Result<&str> {
	let raw = &header[..100];
	let len = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
	std::str::from_utf8(&raw[..len])
		.map_err(|_| Error::data_loss("tar member name is not valid UTF-8"))
}

fn le16(bytes: &[u8]) -> u16 {
	u16::from_le_bytes([bytes[0], bytes[1]])
}

fn le64(bytes: &[u8]) -> Result<u64> {
	Ok(u64::from_le_bytes(
		bytes
			.get(..8)
			.ok_or_else(|| Error::data_loss("zip64 record is truncated"))?
			.try_into()
			.map_err(|_| Error::data_loss("zip64 record is truncated"))?,
	))
}

fn le32(bytes: &[u8]) -> u32 {
	u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
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

#[cfg(test)]
#[path = "../../test/rs/extract/test_zip.rs"]
mod test_zip;
