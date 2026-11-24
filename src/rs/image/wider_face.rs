//! WIDER FACE detection as a lazy [`oa::data::Dataset`].
//!
//! `WIDER_{train,val,test}.zip` and `wider_face_split.zip` extract
//! through [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `wider_face/1.0.0` per split. `wider_face_*_bbx_gt.txt` lists each
//! image path, its box count, then `x1 y1 w h attrs(6)` rows; rows
//! flagged `invalid` are skipped and boxes convert to
//! `[x1, y1, x2, y2]` in native pixels. Test ships only a filelist, so
//! its samples slice to `[0, 5]` boxes. All images carry the single
//! class `face`. Only the `[T, 5]` F32 box matrix is resident;
//! [`oa::data::Dataset::get_item`] lazily decodes one `[3, H, W]`
//! image.

use std::path::Path;

use oa::{Error, Result};

use crate::image::folder::DetectionIndex;

#[path = "wider_face.gen.rs"]
mod generated;

pub(crate) use generated::WIDER_FACE_PACK;
pub use generated::{WiderFace, WiderFaceConfig, WiderFaceSplit};

/// Index `root` (one extracted split tree): the split's `_bbx_gt.txt`
/// or `_filelist.txt` drives path order relative to the image root.
/// Classes are the single `face` label.
fn index(root: &Path, config: &WiderFaceConfig) -> Result<DetectionIndex> {
	let (image_dir, list) = match config.split {
		WiderFaceSplit::Train => (
			root.join("WIDER_train/images"),
			root.join("wider_face_split/wider_face_train_bbx_gt.txt"),
		),
		WiderFaceSplit::Validation => (
			root.join("WIDER_val/images"),
			root.join("wider_face_split/wider_face_val_bbx_gt.txt"),
		),
		WiderFaceSplit::Test => (
			root.join("WIDER_test/images"),
			root.join("wider_face_split/wider_face_test_filelist.txt"),
		),
	};
	let text = std::fs::read_to_string(&list)
		.map_err(|error| Error::io("read wider face split list", error))?;
	let mut lines = text.lines().map(str::trim).filter(|line| !line.is_empty());
	let mut paths = Vec::new();
	let mut boxes = Vec::new();
	let mut box_lengths = Vec::new();
	while let Some(name) = lines.next() {
		paths.push(image_dir.join(name));
		let before = boxes.len();
		if config.split != WiderFaceSplit::Test {
			let count: usize = lines
				.next()
				.ok_or_else(|| Error::data_loss("wider face list misses box count"))?
				.parse()
				.map_err(|_| Error::data_loss("wider face box count is not a number"))?;
			// The upstream format appends a dummy zero row to zero-box entries.
			let count = if count == 0 {
				lines.next();
				0
			} else {
				count
			};
			for _ in 0..count {
				let line = lines
					.next()
					.ok_or_else(|| Error::data_loss("wider face list truncates box rows"))?;
				parse_box(line, &mut boxes)?;
			}
		}
		box_lengths.push(u32::try_from(boxes.len() - before).unwrap_or(u32::MAX));
	}
	Ok(DetectionIndex {
		paths,
		boxes,
		box_lengths,
		classes: vec!["face".to_owned()],
	})
}

/// One ground-truth row is `x1 y1 w h blur expression illumination
/// invalid occlusion pose`; invalid rows are skipped.
fn parse_box(line: &str, out: &mut Vec<[f32; 5]>) -> Result<()> {
	let field = |index: usize| -> Result<f32> {
		line
			.split_whitespace()
			.nth(index)
			.ok_or_else(|| Error::data_loss("wider face box row is truncated"))?
			.parse()
			.map_err(|_| Error::data_loss("wider face box field is not a number"))
	};
	if field(7)? != 0.0 {
		return Ok(());
	}
	let (x, y, w, h) = (field(0)?, field(1)?, field(2)?, field(3)?);
	out.push([x, y, x + w, y + h, 0.0]);
	Ok(())
}
