//! Minimal HDF5 reader for pack datasets.
//!
//! Scope is deliberately narrow — enough to walk a version-0 superblock,
//! resolve a dataset path through symbol-table groups (v1 B-tree nodes +
//! local heaps), and read a dataset stored contiguous or chunked
//! (optionally through the deflate pipeline filter). Compact/dense groups
//! and object headers beyond version 1 fail with a clear `data_loss`
//! naming the unsupported feature.
//!
//! Datasets are read straight from the decompressed file: callers
//! materialize only the rows they ask for, which keeps the streaming
//! decode contract for GiB-scale members.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use oa::{Error, Result};

const SUPERBLOCK_SIG: &[u8; 8] = b"\x89HDF\r\n\x1a\n";
const UNDEFINED: u64 = u64::MAX;

/// One resolved HDF5 dataset: shape plus raw little-endian element bytes.
pub(crate) struct Dataset {
	/// Dataspace dimensions, outermost first.
	pub shape: Vec<u64>,
	/// Element size in bytes.
	pub element_size: u64,
	/// True when the datatype message declares an unsigned fixed-point
	/// class (the only class this reader decodes).
	pub unsigned: bool,
	storage: Storage,
}

enum Storage {
	Contiguous {
		address: u64,
	},
	/// Chunked storage: v1 B-tree root address, chunk dims (one per
	/// dataspace dim plus an element-size slot), deflate flag.
	Chunked {
		btree: u64,
		dims: Vec<u64>,
		deflated: bool,
	},
}

/// A layout message before the filter pipeline is known.
enum RawLayout {
	Contiguous { address: u64 },
	Chunked { btree: u64, dims: Vec<u64> },
}

struct Reader {
	file: File,
	offset_size: usize,
}

impl Reader {
	fn read_at(&mut self, offset: u64, buffer: &mut [u8]) -> Result<()> {
		self
			.file
			.seek(SeekFrom::Start(offset))
			.and_then(|_| self.file.read_exact(buffer))
			.map_err(|error| Error::io("read hdf5 block", error))
	}

	fn u64_at(&mut self, offset: u64) -> Result<u64> {
		let mut bytes = [0u8; 8];
		self.read_at(offset, &mut bytes[..self.offset_size.min(8)])?;
		Ok(u64::from_le_bytes(bytes))
	}
}

/// Resolve dataset `path` (e.g. `"x"` or `"group/sub/x"`) inside the HDF5
/// file at `file` and return its shape plus storage descriptor.
///
/// # Errors
///
/// Returns I/O errors, `data_loss` for malformed or unsupported HDF5
/// structures (non-symbol-table groups, non-integer datatypes), and
/// `invalid_argument` when `path` does not resolve.
pub(crate) fn open_dataset(file: &Path, path: &str) -> Result<Dataset> {
	let mut reader = Reader {
		file: File::open(file).map_err(|error| Error::io("open hdf5 file", error))?,
		offset_size: 8,
	};
	let root = superblock(&mut reader)?;
	let mut object = root;
	for segment in path.trim_matches('/').split('/') {
		if segment.is_empty() {
			continue;
		}
		object = group_link(&mut reader, object, segment)?;
	}
	dataset_header(&mut reader, object)
}

/// Read up to `rows` outermost slices of `dataset` from `file` into a
/// flat byte vec. `rows == u64::MAX` reads the whole dataset. Contiguous
/// storage is sliced on disk; chunked storage walks the chunk B-tree and
/// stops as soon as enough leading rows are covered.
///
/// # Errors
///
/// Returns I/O and inflate errors.
pub(crate) fn read_dataset_rows(file: &Path, dataset: &Dataset, rows: u64) -> Result<Vec<u8>> {
	let mut reader = File::open(file).map_err(|error| Error::io("open hdf5 file", error))?;
	let row_elems: u64 = dataset.shape[1..].iter().product::<u64>().max(1);
	let take_rows = rows.min(dataset.shape.first().copied().unwrap_or(0));
	let take_bytes = usize::try_from(take_rows.saturating_mul(row_elems) * dataset.element_size)
		.map_err(|_| Error::data_loss("hdf5 row slice overflows usize"))?;
	match &dataset.storage {
		Storage::Contiguous { address } => {
			let mut data = vec![0u8; take_bytes];
			reader
				.seek(SeekFrom::Start(*address))
				.and_then(|_| reader.read_exact(&mut data))
				.map_err(|error| Error::io("read hdf5 dataset", error))?;
			Ok(data)
		}
		Storage::Chunked {
			btree,
			dims,
			deflated,
		} => {
			let chunk_rows = dims.first().copied().unwrap_or(1).max(1);
			let chunk_bytes = usize::try_from(dims.iter().product::<u64>() * dataset.element_size)
				.map_err(|_| Error::data_loss("hdf5 chunk size overflows usize"))?;
			let needed = take_rows.div_ceil(chunk_rows);
			let mut chunks =
				Vec::with_capacity(usize::try_from(needed.min(1 << 12)).unwrap_or(usize::MAX));
			walk_chunks(
				&mut reader,
				*btree,
				dims.len(),
				&mut |size, mask, address| {
					if chunks.len() as u64 >= needed {
						return Err(Error::cancelled("hdf5 row cap reached"));
					}
					chunks.push((size, mask, address));
					Ok(())
				},
			)?;
			let mut data = Vec::with_capacity(chunks.len() * chunk_bytes);
			for (size, mask, address) in chunks {
				let raw = read_chunk(&mut reader, address, size, mask, *deflated)?;
				data.extend_from_slice(&raw[..raw.len().min(chunk_bytes)]);
			}
			data.truncate(take_bytes);
			Ok(data)
		}
	}
}

/// Read one chunk payload at `address`; `mask & 1` means deflate.
fn read_chunk(
	reader: &mut File,
	address: u64,
	size: u64,
	mask: u64,
	deflated: bool,
) -> Result<Vec<u8>> {
	let mut raw = vec![
		0u8;
		usize::try_from(size)
			.map_err(|_| Error::data_loss("hdf5 chunk extent overflows usize"))?
	];
	reader
		.seek(SeekFrom::Start(address))
		.and_then(|_| reader.read_exact(&mut raw))
		.map_err(|error| Error::io("read hdf5 chunk", error))?;
	if mask & 1 == 0 || !deflated {
		return Ok(raw);
	}
	let mut out = Vec::new();
	flate2::read::ZlibDecoder::new(&raw[..])
		.read_to_end(&mut out)
		.map_err(|error| Error::io("inflate hdf5 chunk", error))?;
	Ok(out)
}

/// Walk a chunked-storage v1 B-tree (node type 1) depth-first, invoking
/// `visit(chunk_size, filter_mask, data_address)` per leaf record in key
/// order. `Cancelled` from `visit` stops the walk and is swallowed.
fn walk_chunks(
	reader: &mut File,
	node: u64,
	ndims: usize,
	visit: &mut impl FnMut(u64, u64, u64) -> Result<()>,
) -> Result<()> {
	let mut head = [0u8; 24];
	reader
		.seek(SeekFrom::Start(node))
		.and_then(|_| reader.read_exact(&mut head))
		.map_err(|error| Error::io("read hdf5 chunk b-tree", error))?;
	if &head[..4] != b"TREE" || head[4] != 1 {
		return Err(Error::data_loss("hdf5 chunk b-tree signature missing"));
	}
	let level = head[5];
	let entries = u16::from_le_bytes(head[6..8].try_into().unwrap_or_default()) as u64;
	// Chunk node key: chunk size(4) + filter mask(4) + scaled
	// offsets(8 × ndims); records are (key, child) pairs.
	let key = 8 + 8 * ndims as u64;
	for index in 0..entries {
		let record = node + 24 + index * (key + 8);
		let mut kbuf = vec![0u8; key as usize];
		reader
			.seek(SeekFrom::Start(record))
			.and_then(|_| reader.read_exact(&mut kbuf))
			.map_err(|error| Error::io("read hdf5 chunk record", error))?;
		let size = u32::from_le_bytes(kbuf[..4].try_into().unwrap_or_default()) as u64;
		let mask = u32::from_le_bytes(kbuf[4..8].try_into().unwrap_or_default()) as u64;
		let child = {
			let mut bytes = [0u8; 8];
			reader
				.seek(SeekFrom::Start(record + key))
				.and_then(|_| reader.read_exact(&mut bytes))
				.map_err(|error| Error::io("read hdf5 chunk child", error))?;
			u64::from_le_bytes(bytes)
		};
		if child == 0 || child == UNDEFINED {
			continue;
		}
		let result = if level > 0 {
			walk_chunks(reader, child, ndims, visit)
		} else {
			visit(size, mask, child)
		};
		match result {
			Err(error) if error.kind() == oa::ErrorKind::Cancelled => return Ok(()),
			result => result?,
		}
	}
	Ok(())
}

/// Parse a version-0 superblock and return the root group's object header
/// address.
fn superblock(reader: &mut Reader) -> Result<u64> {
	let mut signature = [0u8; 8];
	reader.read_at(0, &mut signature)?;
	if &signature != SUPERBLOCK_SIG {
		return Err(Error::data_loss("hdf5 signature missing"));
	}
	let mut header = [0u8; 56];
	reader.read_at(8, &mut header)?;
	if header[0] != 0 {
		return Err(Error::data_loss(
			"hdf5 superblock versions above 0 are unsupported",
		));
	}
	let offset_size = header[5] as usize;
	if offset_size != 8 {
		return Err(Error::data_loss("hdf5 non-8-byte offsets are unsupported"));
	}
	reader.offset_size = offset_size;
	// Root group symbol table entry follows the fixed header and the
	// base/free/eof/driver offsets: name-offset, object-header address,
	// cache type, reserved, scratch.
	let entry = 24 + 4 * offset_size;
	let object = reader.u64_at((entry + offset_size) as u64)?;
	if object == UNDEFINED {
		return Err(Error::data_loss("hdf5 root group address is undefined"));
	}
	Ok(object)
}

/// A v1 object header message: type plus its payload.
struct Message {
	kind: u16,
	data: Vec<u8>,
}

/// Read every message in the v1 object header at `address`, following
/// continuation blocks.
fn object_header(reader: &mut Reader, address: u64) -> Result<Vec<Message>> {
	let mut head = [0u8; 16];
	reader.read_at(address, &mut head)?;
	if head[0] != 1 {
		return Err(Error::data_loss(
			"hdf5 object header versions above 1 are unsupported",
		));
	}
	let count = u16::from_le_bytes(head[2..4].try_into().unwrap_or_default()) as usize;
	let header_size = u32::from_le_bytes(head[8..12].try_into().unwrap_or_default()) as u64;
	let mut messages = Vec::with_capacity(count);
	let mut blocks = vec![(address + 16, header_size)];
	while let Some((start, length)) = blocks.pop() {
		let mut cursor = start;
		let limit = start + length;
		while cursor + 8 <= limit {
			let mut meta = [0u8; 8];
			reader.read_at(cursor, &mut meta)?;
			let kind = u16::from_le_bytes(meta[0..2].try_into().unwrap_or_default());
			let size = u16::from_le_bytes(meta[2..4].try_into().unwrap_or_default()) as u64;
			let payload = cursor + 8;
			let mut data = vec![0u8; size as usize];
			reader.read_at(payload, &mut data)?;
			if kind == 0x0010 {
				let block = u64::from_le_bytes(data[..8].try_into().unwrap_or_default());
				let length = u64::from_le_bytes(data[8..16].try_into().unwrap_or_default());
				blocks.push((block, length));
			} else if kind != 0 {
				messages.push(Message { kind, data });
			}
			cursor = payload + size;
		}
	}
	Ok(messages)
}

/// Resolve `name` inside the symbol-table group whose object header is at
/// `address`, returning the linked object header address.
fn group_link(reader: &mut Reader, address: u64, name: &str) -> Result<u64> {
	let messages = object_header(reader, address)?;
	let table = messages
		.iter()
		.find(|message| message.kind == 0x0011)
		.ok_or_else(|| Error::data_loss("hdf5 group lacks a symbol table message"))?;
	let btree = u64::from_le_bytes(table.data[..8].try_into().unwrap_or_default());
	let heap = u64::from_le_bytes(table.data[8..16].try_into().unwrap_or_default());
	let heap_data = heap_data_segment(reader, heap)?;
	btree_leaf(reader, btree, name, heap_data)
}

/// Local heap data segment address.
fn heap_data_segment(reader: &mut Reader, heap: u64) -> Result<u64> {
	let mut head = [0u8; 32];
	reader.read_at(heap, &mut head)?;
	if &head[..4] != b"HEAP" {
		return Err(Error::data_loss("hdf5 local heap signature missing"));
	}
	Ok(u64::from_le_bytes(
		head[24..32].try_into().unwrap_or_default(),
	))
}

/// Walk the group v1 B-tree (signature `TREE`, node type 0) in key order,
/// scanning each leaf symbol table node for `name`. Node records are
/// (key, child) pairs — `entries` records with a trailing boundary key.
fn btree_leaf(reader: &mut Reader, node: u64, name: &str, heap: u64) -> Result<u64> {
	let mut head = [0u8; 24];
	reader.read_at(node, &mut head)?;
	if &head[..4] != b"TREE" || head[4] != 0 {
		return Err(Error::data_loss("hdf5 group b-tree signature missing"));
	}
	let level = head[5];
	let entries = u16::from_le_bytes(head[6..8].try_into().unwrap_or_default()) as u64;
	for index in 0..entries {
		let child = reader.u64_at(node + 24 + index * 16 + 8)?;
		if child == 0 || child == UNDEFINED {
			continue;
		}
		if level > 0 {
			if let Ok(found) = btree_leaf(reader, child, name, heap) {
				return Ok(found);
			}
		} else if let Some(object) = snod_link(reader, child, name, heap)? {
			return Ok(object);
		}
	}
	Err(Error::invalid_argument(format!(
		"hdf5 group link {name} not found"
	)))
}

/// Scan a symbol table node (`SNOD`) for `name`; each entry is
/// name-offset(8) + object-address(8) + cache-type(4) + reserved(4) +
/// scratch(16).
fn snod_link(reader: &mut Reader, snod: u64, name: &str, heap: u64) -> Result<Option<u64>> {
	let mut head = [0u8; 8];
	reader.read_at(snod, &mut head)?;
	if &head[..4] != b"SNOD" {
		return Err(Error::data_loss("hdf5 snod signature missing"));
	}
	let entries = u16::from_le_bytes(head[6..8].try_into().unwrap_or_default()) as u64;
	for index in 0..entries {
		let entry = snod + 8 + index * 40;
		let name_offset = reader.u64_at(entry)?;
		let object = reader.u64_at(entry + 8)?;
		let mut bytes = [0u8; 256];
		reader.read_at(heap + name_offset, &mut bytes)?;
		let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
		let link = std::str::from_utf8(&bytes[..end])
			.map_err(|_| Error::data_loss("hdf5 link name is not valid UTF-8"))?;
		if link == name {
			return Ok(Some(object));
		}
	}
	Ok(None)
}

/// Extract shape, datatype, and storage layout from a dataset object
/// header's messages.
fn dataset_header(reader: &mut Reader, address: u64) -> Result<Dataset> {
	let messages = object_header(reader, address)?;
	let mut shape = None;
	let mut element_size = None;
	let mut unsigned = None;
	let mut layout = None;
	let mut deflated = false;
	for message in &messages {
		match message.kind {
			0x0001 => shape = Some(dataspace(&message.data)?),
			0x0003 => {
				let (size, uns) = datatype(&message.data)?;
				element_size = Some(size);
				unsigned = Some(uns);
			}
			0x0008 => layout = Some(data_layout(&message.data)?),
			0x000B => deflated = filter_deflate(&message.data)?,
			_ => {}
		}
	}
	let shape = shape.ok_or_else(|| Error::data_loss("hdf5 dataset lacks dataspace"))?;
	let element_size = element_size.ok_or_else(|| Error::data_loss("hdf5 dataset lacks datatype"))?;
	let unsigned = unsigned.unwrap_or(false);
	let layout = layout.ok_or_else(|| Error::data_loss("hdf5 dataset lacks data layout"))?;
	let storage = match layout {
		RawLayout::Contiguous { address } => {
			if deflated {
				return Err(Error::data_loss(
					"hdf5 filtered contiguous storage is unsupported",
				));
			}
			Storage::Contiguous { address }
		}
		RawLayout::Chunked { btree, dims } => Storage::Chunked {
			btree,
			dims,
			deflated,
		},
	};
	Ok(Dataset {
		shape,
		element_size,
		unsigned,
		storage,
	})
}

/// Dataspace message v1: version, rank, flags, reserved, dims.
fn dataspace(data: &[u8]) -> Result<Vec<u64>> {
	if data.len() < 8 {
		return Err(Error::data_loss("hdf5 dataspace message truncated"));
	}
	let rank = data[1] as usize;
	let mut shape = Vec::with_capacity(rank);
	for index in 0..rank {
		let at = 8 + index * 8;
		if data.len() < at + 8 {
			return Err(Error::data_loss("hdf5 dataspace dims truncated"));
		}
		shape.push(u64::from_le_bytes(
			data[at..at + 8].try_into().unwrap_or_default(),
		));
	}
	Ok(shape)
}

/// Datatype message: class-and-version byte, class bit field, size.
/// Returns `(element_size, unsigned)` for fixed-point class 0.
fn datatype(data: &[u8]) -> Result<(u64, bool)> {
	if data.len() < 8 {
		return Err(Error::data_loss("hdf5 datatype message truncated"));
	}
	let class = data[0] & 0x0f;
	if class != 0 {
		return Err(Error::data_loss(
			"hdf5 non-fixed-point datatypes are unsupported",
		));
	}
	let unsigned = data[1] & 0x08 == 0;
	let size = u32::from_le_bytes(data[4..8].try_into().unwrap_or_default()) as u64;
	Ok((size, unsigned))
}

/// Data layout message v3: version, class. Class 1 = contiguous
/// (address + size); class 2 = chunked (ndims, b-tree address, chunk
/// dims, element size). Compact storage is rejected.
fn data_layout(data: &[u8]) -> Result<RawLayout> {
	if data.len() < 2 {
		return Err(Error::data_loss("hdf5 data layout message truncated"));
	}
	if data[0] != 3 {
		return Err(Error::data_loss(
			"hdf5 data layout versions other than 3 are unsupported",
		));
	}
	match data[1] {
		1 => {
			if data.len() < 18 {
				return Err(Error::data_loss("hdf5 contiguous layout truncated"));
			}
			let address = u64::from_le_bytes(data[2..10].try_into().unwrap_or_default());
			Ok(RawLayout::Contiguous { address })
		}
		2 => {
			let ndims = data[2] as usize;
			if data.len() < 11 + ndims * 4 {
				return Err(Error::data_loss("hdf5 chunked layout truncated"));
			}
			let btree = u64::from_le_bytes(data[3..11].try_into().unwrap_or_default());
			let mut dims = Vec::with_capacity(ndims);
			for index in 0..ndims {
				let at = 11 + index * 4;
				dims.push(u32::from_le_bytes(data[at..at + 4].try_into().unwrap_or_default()) as u64);
			}
			Ok(RawLayout::Chunked { btree, dims })
		}
		_ => Err(Error::data_loss("hdf5 compact layouts are unsupported")),
	}
}

/// Filter pipeline message: true when the deflate filter (id 1) is
/// installed. Other filters are rejected.
fn filter_deflate(data: &[u8]) -> Result<bool> {
	if data.len() < 8 {
		return Ok(false);
	}
	let count = u16::from_le_bytes(data[2..4].try_into().unwrap_or_default()) as usize;
	let mut cursor = 8;
	let mut deflated = false;
	for _ in 0..count {
		if data.len() < cursor + 8 {
			return Err(Error::data_loss("hdf5 filter entry truncated"));
		}
		let id = u16::from_le_bytes(data[cursor..cursor + 2].try_into().unwrap_or_default());
		let name_len =
			u16::from_le_bytes(data[cursor + 2..cursor + 4].try_into().unwrap_or_default()) as usize;
		let params =
			u16::from_le_bytes(data[cursor + 6..cursor + 8].try_into().unwrap_or_default()) as usize;
		if id == 1 {
			deflated = true;
		} else {
			return Err(Error::data_loss(format!(
				"hdf5 pipeline filter {id} is unsupported"
			)));
		}
		cursor += 8 + name_len + params * 4;
	}
	Ok(deflated)
}
