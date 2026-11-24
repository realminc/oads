//! Lazy image-folder indexing shared by the image-classification packs.
//!
//! Packs ship their samples as one raster file per class directory (or a
//! per-pack manifest naming the class). [`ImageIndex`] records the sorted
//! `(path, label)` pairing once after extraction; loaders upload only the
//! `[N]` labels matrix and decode each `[1, C, H, W]` sample lazily
//! through [`oa::image`], keeping multi-gigabyte corpora off the device
//! until accessed.

use std::path::{Path, PathBuf};

use oa::image::{Image, ImageFormat};
use oa::{Engine, Error, Result};

use crate::core::decode::MatrixData;
use crate::core::extract;

/// Raster extensions [`oa::image`] decodes.
pub(crate) const IMAGE_EXTENSIONS: [&str; 8] =
	["jpg", "jpeg", "png", "bmp", "webp", "tga", "tif", "tiff"];

/// One extracted image folder: sorted member paths, the parallel label
/// vector, and the class-name table labels index into.
pub(crate) struct ImageIndex {
	pub paths: Vec<PathBuf>,
	pub labels: Vec<u16>,
	pub classes: Vec<String>,
}

impl ImageIndex {
	/// Index `root` where every immediate subdirectory is one class.
	/// Classes sort by directory name; members sort by path within and
	/// across classes, so `(paths, labels)` are deterministic.
	pub(crate) fn class_dirs(root: &Path) -> Result<Self> {
		let mut classes = Vec::new();
		for entry in read_dir(root)? {
			let path = entry.path();
			if path.is_dir() {
				classes.push(entry.file_name().to_string_lossy().into_owned());
			}
		}
		classes.sort();
		let mut paths = Vec::new();
		let mut labels = Vec::new();
		for (label, class) in classes.iter().enumerate() {
			for file in image_files(&root.join(class))? {
				paths.push(file);
				labels.push(label_index(label)?);
			}
		}
		Ok(Self {
			paths,
			labels,
			classes,
		})
	}

	/// Index image files under `root` recursively, deriving the class
	/// table from `label_of`, which maps each file path to a class name
	/// (`None` skips the file — thumbnails, manifests, corrupt members).
	/// Classes sort by name; members sort by path.
	pub(crate) fn scan(root: &Path, label_of: impl Fn(&Path) -> Option<String>) -> Result<Self> {
		let mut files = Vec::new();
		collect_images(root, &mut files)?;
		files.sort();
		let mut classes = Vec::new();
		let mut labeled = Vec::with_capacity(files.len());
		for path in files {
			if let Some(name) = label_of(&path) {
				if !classes.contains(&name) {
					classes.push(name.clone());
				}
				labeled.push((path, name));
			}
		}
		classes.sort();
		let mut paths = Vec::with_capacity(labeled.len());
		let mut labels = Vec::with_capacity(labeled.len());
		for (path, name) in labeled {
			let label = classes.iter().position(|class| *class == name);
			paths.push(path);
			labels.push(label_index(label.ok_or_else(|| {
				Error::internal("derived image class escaped the class table")
			})?)?);
		}
		Ok(Self {
			paths,
			labels,
			classes,
		})
	}

	/// `[N]` F32 label matrix for [`Decoded`](crate::core::decode::Decoded).
	pub(crate) fn labels(&self) -> Result<MatrixData> {
		let values = self
			.labels
			.iter()
			.map(|&label| f32::from(label))
			.collect::<Vec<_>>();
		Ok(MatrixData {
			name: "labels",
			shape: vec![self.labels.len()],
			values,
		})
	}

	/// Newline-joined root-relative paths in label order for
	/// `Decoded::text`, letting foreign bindings resolve members without
	/// reimplementing the scan ordering.
	pub(crate) fn paths_text(&self, root: &Path) -> Result<String> {
		let mut text = String::new();
		for path in &self.paths {
			let relative = path
				.strip_prefix(root)
				.map_err(|_| Error::internal("image index path escapes its extracted root"))?;
			text.push_str(&relative.to_string_lossy());
			text.push('\n');
		}
		Ok(text)
	}
}

/// One indexed object-detection split: sorted image paths, the flat
/// `[x1, y1, x2, y2, class]` box rows across all paths, the per-path box
/// counts, and the class-name table box labels index into.
///
/// Box coordinates are native image pixels in the upstream's frame;
/// `class` is a float-encoded index into `classes`.
pub(crate) struct DetectionIndex {
	pub paths: Vec<PathBuf>,
	pub boxes: Vec<[f32; 5]>,
	pub box_lengths: Vec<u32>,
	pub classes: Vec<String>,
}

impl DetectionIndex {
	/// `[T, 5]` F32 box matrix for [`Decoded`](crate::core::decode::Decoded).
	pub(crate) fn boxes(&self) -> Result<MatrixData> {
		if self.boxes.len() != self.box_lengths.iter().map(|&n| n as usize).sum::<usize>() {
			return Err(Error::internal(
				"detection box rows do not match box lengths",
			));
		}
		let values = self
			.boxes
			.iter()
			.flat_map(|row| row.iter().copied())
			.collect::<Vec<_>>();
		Ok(MatrixData {
			name: "boxes",
			shape: vec![self.boxes.len(), 5],
			values,
		})
	}

	/// `[N]` F32 per-path box counts for
	/// [`Decoded`](crate::core::decode::Decoded).
	pub(crate) fn box_lengths(&self) -> Result<MatrixData> {
		let values = self
			.box_lengths
			.iter()
			.map(|&count| count as f32)
			.collect::<Vec<_>>();
		Ok(MatrixData {
			name: "box_lengths",
			shape: vec![self.box_lengths.len()],
			values,
		})
	}

	/// `[N + 1]` host offsets: `offsets[i]..offsets[i + 1]` spans the
	/// box rows of path `i`.
	pub(crate) fn offsets(&self) -> Vec<i64> {
		let mut offsets = Vec::with_capacity(self.box_lengths.len() + 1);
		offsets.push(0);
		for &count in &self.box_lengths {
			offsets.push(offsets.last().copied().unwrap_or(0) + i64::from(count));
		}
		offsets
	}

	/// Newline-joined root-relative paths in index order for
	/// `Decoded::text`, letting foreign bindings resolve members without
	/// reimplementing the scan ordering.
	pub(crate) fn paths_text(&self, root: &Path) -> Result<String> {
		let mut text = String::new();
		for path in &self.paths {
			let relative = path
				.strip_prefix(root)
				.map_err(|_| Error::internal("image index path escapes its extracted root"))?;
			text.push_str(&relative.to_string_lossy());
			text.push('\n');
		}
		Ok(text)
	}
}

/// Extract every regular member of `archive` — zip, tar.gz, tar.bz2, or
/// plain tar, selected by magic bytes — into `dest`, flattening nothing
/// and rejecting members that would escape `dest`.
pub(crate) fn extract_images(archive: &Path, dest: &Path) -> Result<()> {
	use std::io::Read;
	let mut magic = [0u8; 4];
	std::fs::File::open(archive)
		.and_then(|mut file| file.read_exact(&mut magic))
		.map_err(|error| Error::io("read archive magic", error))?;
	let write = move |name: &str, data: &[u8], dest: &Path| -> Result<()> {
		let member = safe_member(dest, name)?;
		if let Some(parent) = member.parent() {
			std::fs::create_dir_all(parent).map_err(|error| Error::io("create member dir", error))?;
		}
		std::fs::write(&member, data).map_err(|error| Error::io("write member", error))
	};
	match &magic {
		[0x50, 0x4B, ..] => extract::walk_zip_file(
			archive,
			|_| true,
			|name, data| {
				if name.ends_with('/') {
					return Ok(());
				}
				write(name, data, dest)
			},
		),
		[0x1F, 0x8B, ..] => extract::walk_tar_gz(archive, |name, data| write(name, data, dest)),
		[b'B', b'Z', b'h', ..] => extract::walk_tar_bz2(archive, |name, data| write(name, data, dest)),
		_ => {
			if is_tar(archive)? {
				extract::walk_tar_file(archive, |name, data| write(name, data, dest))
			} else {
				Err(Error::data_loss(format!(
					"{} is not a zip or tar archive",
					archive.display()
				)))
			}
		}
	}
}

/// Decode `path` to a `[1, C, H, W]` F32 device image at its native size
/// through [`oa::image::decode_file_on`].
pub(crate) fn decode(engine: &Engine, path: &Path) -> Result<Image> {
	oa::image::decode_file_on(engine, path, ImageFormat::Rgb)
}

fn is_tar(path: &Path) -> Result<bool> {
	use std::io::Read;
	let mut file = std::fs::File::open(path).map_err(|error| Error::io("open archive", error))?;
	let mut block = [0u8; 512];
	if file
		.read(&mut block)
		.map_err(|error| Error::io("read archive", error))?
		< 512
	{
		return Ok(false);
	}
	if block.iter().all(|byte| *byte == 0) {
		return Ok(false);
	}
	if &block[257..262] == b"ustar" {
		return Ok(true);
	}
	// V7 headers carry no magic; validate the stored header checksum
	// (bytes 148..156 read as spaces when summed).
	let digits = block[148..156]
		.iter()
		.take_while(|byte| byte.is_ascii_digit() || **byte == b' ');
	let stored = digits.fold(0u64, |sum, byte| {
		if byte.is_ascii_digit() {
			sum * 8 + u64::from(byte - b'0')
		} else {
			sum
		}
	});
	let sum: u64 = block
		.iter()
		.enumerate()
		.map(|(index, byte)| {
			if (148..156).contains(&index) {
				u64::from(b' ')
			} else {
				u64::from(*byte)
			}
		})
		.sum();
	Ok(stored == sum)
}

fn safe_member(dest: &Path, name: &str) -> Result<PathBuf> {
	let member = dest.join(name);
	if name.split('/').any(|part| part == "..") || member.strip_prefix(dest).is_err() {
		return Err(Error::data_loss(format!(
			"archive member {name} escapes the extract root"
		)));
	}
	Ok(member)
}

fn read_dir(root: &Path) -> Result<Vec<std::fs::DirEntry>> {
	let entries = std::fs::read_dir(root)
		.map_err(|error| Error::io("read image class dir", error))?
		.collect::<std::result::Result<Vec<_>, _>>()
		.map_err(|error| Error::io("read image class dir", error))?;
	Ok(entries)
}

fn image_files(dir: &Path) -> Result<Vec<PathBuf>> {
	let mut files = Vec::new();
	for entry in read_dir(dir)? {
		let path = entry.path();
		if path.is_file() && is_image(&path) {
			files.push(path);
		}
	}
	files.sort();
	Ok(files)
}

/// Recursively collect supported image files under `dir`, unsorted.
pub(crate) fn collect_images(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
	for entry in read_dir(dir)? {
		let path = entry.path();
		if path.is_dir() {
			collect_images(&path, out)?;
		} else if is_image(&path) {
			out.push(path);
		}
	}
	Ok(())
}

fn label_index(label: usize) -> Result<u16> {
	u16::try_from(label)
		.map_err(|_| Error::out_of_range("image class count exceeds the u16 label encoding"))
}

fn is_image(path: &Path) -> bool {
	path
		.extension()
		.and_then(|ext| ext.to_str())
		.map(|ext| {
			let ext = ext.to_ascii_lowercase();
			IMAGE_EXTENSIONS.contains(&ext.as_str())
		})
		.unwrap_or(false)
}
