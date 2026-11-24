//! TFRecord stream walking and the minimal protobuf reads loaders need.
//!
//! A `.tfrecord` file is a sequence of
//! `u64 length | u32 length-crc | payload | u32 payload-crc` frames where
//! each payload is a serialized `tf.train.Example` protobuf. Pack file
//! integrity is already covered by `sha256` verification in
//! [`ensure_pack`](crate::core::pack::ensure_pack), so the per-frame CRC
//! fields are consumed but not recomputed.
//!
//! The protobuf surface is read with the smallest possible scanner:
//! `Example.features` is a `map<string, Feature>` and only the named
//! feature's bytes values are extracted — every other field is skipped
//! without allocation. This keeps multi-GB shards on the streaming decode
//! contract: one record payload lives in memory at a time.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use oa::{Error, Result};

/// Invoke `visit` once per record payload in the TFRecord stream at `path`.
/// Returning [`Error::cancelled`] from `visit` stops the walk early and is
/// swallowed; any other error propagates.
///
/// # Errors
///
/// Returns I/O errors opening or reading `path`, `data_loss` for truncated
/// or malformed frames, plus any error `visit` returns.
pub(crate) fn walk_tfrecord(path: &Path, mut visit: impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
	let file = File::open(path).map_err(|error| Error::io("open tfrecord shard", error))?;
	let mut reader = BufReader::with_capacity(1 << 20, file);
	let mut header = [0u8; 12];
	loop {
		match reader.read_exact(&mut header) {
			Ok(()) => {}
			Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
			Err(error) => return Err(Error::io("read tfrecord header", error)),
		}
		let length = u64::from_le_bytes(header[..8].try_into().unwrap_or_default());
		let length = usize::try_from(length)
			.map_err(|_| Error::data_loss("tfrecord payload length overflows usize"))?;
		let mut payload = vec![0u8; length];
		reader
			.read_exact(&mut payload)
			.map_err(|error| Error::io("read tfrecord payload", error))?;
		let mut crc = [0u8; 4];
		reader
			.read_exact(&mut crc)
			.map_err(|error| Error::io("read tfrecord crc", error))?;
		match visit(&payload) {
			Err(error) if error.kind() == oa::ErrorKind::Cancelled => return Ok(()),
			result => result?,
		}
	}
}

/// Return the first bytes value of feature `name` in a serialized
/// `tf.train.Example`, or `None` when absent. All other fields are skipped
/// without allocation.
pub(crate) fn example_bytes<'a>(payload: &'a [u8], name: &str) -> Option<&'a [u8]> {
	let mut fields = Fields::new(payload);
	while let Some((field, wire, value)) = fields.next() {
		if field == 1
			&& wire == 2
			&& let Some(found) = features_bytes(value, name)
		{
			return Some(found);
		}
	}
	None
}

/// `Features` message: repeated `map<string, Feature>` entries in field 1.
fn features_bytes<'a>(payload: &'a [u8], name: &str) -> Option<&'a [u8]> {
	let mut fields = Fields::new(payload);
	while let Some((field, wire, value)) = fields.next() {
		if field == 1
			&& wire == 2
			&& let Some(found) = entry_bytes(value, name)
		{
			return Some(found);
		}
	}
	None
}

/// Map entry: field 1 is the key string, field 2 the `Feature` message.
/// Fields may arrive in either order, so both are collected before the key
/// is compared.
fn entry_bytes<'a>(payload: &'a [u8], name: &str) -> Option<&'a [u8]> {
	let mut fields = Fields::new(payload);
	let mut key: Option<&[u8]> = None;
	let mut feature: Option<&[u8]> = None;
	while let Some((field, wire, value)) = fields.next() {
		match (field, wire) {
			(1, 2) => key = Some(value),
			(2, 2) => feature = Some(value),
			_ => {}
		}
	}
	if key == Some(name.as_bytes()) {
		feature.and_then(feature_bytes)
	} else {
		None
	}
}

/// `Feature` message: field 1 holds `BytesList`; other kinds are ignored.
fn feature_bytes(payload: &[u8]) -> Option<&[u8]> {
	let mut fields = Fields::new(payload);
	while let Some((field, wire, value)) = fields.next() {
		if field != 1 || wire != 2 {
			continue;
		}
		let mut list = Fields::new(value);
		while let Some((field, wire, value)) = list.next() {
			if field == 1 && wire == 2 {
				return Some(value);
			}
		}
	}
	None
}

/// Field iterator over a protobuf payload. Every wire type's value is
/// consumed inside `next`; length-delimited fields surface their payload
/// slice and varint/fixed fields surface an empty slice.
struct Fields<'a> {
	payload: &'a [u8],
	offset: usize,
}

impl<'a> Fields<'a> {
	fn new(payload: &'a [u8]) -> Self {
		Self { payload, offset: 0 }
	}

	/// Advance to the next field, returning
	/// `(field_number, wire_type, value)`. `value` carries the payload for
	/// wire type 2 and is empty for wire 0/1/5 (already consumed). Unknown
	/// wire types terminate iteration.
	fn next(&mut self) -> Option<(u64, u64, &'a [u8])> {
		let key = self.varint()?;
		let field = key >> 3;
		let wire = key & 7;
		let value = match wire {
			0 => {
				self.varint()?;
				&[]
			}
			1 => self.take(8)?,
			2 => {
				let length = usize::try_from(self.varint()?).ok()?;
				self.take(length)?
			}
			5 => self.take(4)?,
			_ => return None,
		};
		Some((field, wire, value))
	}

	fn take(&mut self, length: usize) -> Option<&'a [u8]> {
		let end = self.offset.checked_add(length)?;
		if end > self.payload.len() {
			return None;
		}
		let value = &self.payload[self.offset..end];
		self.offset = end;
		Some(value)
	}

	fn varint(&mut self) -> Option<u64> {
		let mut value = 0u64;
		for shift in (0..64).step_by(7) {
			let byte = *self.payload.get(self.offset)?;
			self.offset += 1;
			value |= u64::from(byte & 0x7f) << shift;
			if byte & 0x80 == 0 {
				return Some(value);
			}
		}
		None
	}
}
