//! PASCAL VOC 2012 object detection as a lazy [`oa::data::Dataset`].
//!
//! `VOCtrainval_11-May-2012.tar` extracts through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `voc2012/1.0.0`
//! per split. `ImageSets/Main/{train,val}.txt` supplies the official id
//! list; the 2012 test annotations live behind the PASCAL evaluation
//! server and are not public, so this pack ships train/val only. Each
//! id resolves to `JPEGImages/{id}.jpg` plus `Annotations/{id}.xml`,
//! whose `<object>` entries yield `[xmin, ymin, xmax, ymax, class]`
//! rows in native pixels. Only the `[T, 5]` F32 box matrix is resident;
//! [`oa::data::Dataset::get_item`] lazily decodes one `[3, H, W]`
//! image.

use std::path::Path;

use oa::{Error, Result};

use crate::image::folder::DetectionIndex;

#[path = "voc2012.gen.rs"]
mod generated;

pub(crate) use generated::VOC2012_PACK;
pub use generated::{Voc2012, Voc2012Config, Voc2012Split};

/// Index `root` (one extracted VOCdevkit/VOC2012 tree): the split id
/// list drives path order; every `Annotations/{id}.xml` contributes its
/// object boxes. Classes sort alphabetically for the canonical VOC
/// ordering.
fn index(root: &Path, config: &Voc2012Config) -> Result<DetectionIndex> {
	let list = match config.split {
		Voc2012Split::Train => "train",
		Voc2012Split::Validation => "val",
	};
	let ids = std::fs::read_to_string(root.join("ImageSets/Main").join(format!("{list}.txt")))
		.map_err(|error| Error::io("read voc split list", error))?;
	let mut paths = Vec::new();
	let mut classes = Vec::new();
	let mut boxes = Vec::new();
	let mut box_lengths = Vec::new();
	for id in ids.lines().map(str::trim).filter(|line| !line.is_empty()) {
		paths.push(root.join("JPEGImages").join(format!("{id}.jpg")));
		let text = std::fs::read_to_string(root.join("Annotations").join(format!("{id}.xml")))
			.map_err(|error| Error::io("read voc annotation", error))?;
		let before = boxes.len();
		super::voc2007::parse_annotation(&text, &mut classes, &mut boxes)?;
		box_lengths.push(u32::try_from(boxes.len() - before).unwrap_or(u32::MAX));
	}
	// Classes were collected first-seen; remap box labels into the
	// sorted table for the canonical VOC ordering.
	let mut sorted = classes.clone();
	sorted.sort();
	let remap = classes
		.iter()
		.map(|name| sorted.iter().position(|class| class == name).unwrap_or(0))
		.collect::<Vec<_>>();
	for row in &mut boxes {
		row[4] = remap[row[4] as usize] as f32;
	}
	Ok(DetectionIndex {
		paths,
		boxes,
		box_lengths,
		classes: sorted,
	})
}
