//! PASCAL VOC 2007 object detection as a lazy [`oa::data::Dataset`].
//!
//! `VOCtrainval_06-Nov-2007.tar` and `VOCtest_06-Nov-2007.tar` extract
//! through [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `voc2007/1.0.0` per split. `ImageSets/Main/{train,val,test}.txt`
//! supplies the official id list; each id resolves to
//! `JPEGImages/{id}.jpg` plus `Annotations/{id}.xml`, whose `<object>`
//! entries yield `[xmin, ymin, xmax, ymax, class]` rows in native
//! pixels. Only the `[T, 5]` F32 box matrix is resident;
//! [`oa::data::Dataset::get_item`] lazily decodes one `[3, H, W]` image.

use std::path::{Path, PathBuf};

use oa::{Error, Result};

use crate::image::folder::DetectionIndex;

#[path = "voc2007.gen.rs"]
mod generated;

pub(crate) use generated::VOC2007_PACK;
pub use generated::{Voc2007, Voc2007Config, Voc2007Split};

/// VOC 2007 `<annotation>` XML: one `[xmin, ymin, xmax, ymax, class]`
/// row per `<object>`, classes appended to `classes` in first-seen
/// order (the caller sorts after collection).
pub(crate) fn parse_annotation(
	text: &str,
	classes: &mut Vec<String>,
	out: &mut Vec<[f32; 5]>,
) -> Result<()> {
	let mut rest = text;
	while let Some(start) = rest.find("<object>") {
		let end = rest[start..]
			.find("</object>")
			.ok_or_else(|| Error::data_loss("voc annotation object tag is unclosed"))?
			+ start;
		let object = &rest[start..end];
		let name = tag(object, "name")?;
		let class = match classes.iter().position(|class| class == name) {
			Some(index) => index,
			None => {
				classes.push(name.to_owned());
				classes.len() - 1
			}
		};
		out.push([
			number(object, "xmin")?,
			number(object, "ymin")?,
			number(object, "xmax")?,
			number(object, "ymax")?,
			class as f32,
		]);
		rest = &rest[end..];
	}
	Ok(())
}

fn tag<'a>(text: &'a str, name: &str) -> Result<&'a str> {
	let open = format!("<{name}>");
	let close = format!("</{name}>");
	let start = text
		.find(&open)
		.ok_or_else(|| Error::data_loss(format!("voc annotation misses <{name}>")))?
		+ open.len();
	let end = text[start..]
		.find(&close)
		.ok_or_else(|| Error::data_loss(format!("voc annotation <{name}> is unclosed")))?
		+ start;
	Ok(text[start..end].trim())
}

fn number(text: &str, name: &str) -> Result<f32> {
	tag(text, name)?
		.parse()
		.map_err(|_| Error::data_loss(format!("voc annotation <{name}> is not a number")))
}

/// Index `root` (one extracted VOCdevkit/VOC2007 tree): the split id
/// list drives path order; every `Annotations/{id}.xml` contributes its
/// object boxes. Classes sort alphabetically for the canonical VOC
/// ordering.
fn index(root: &Path, config: &Voc2007Config) -> Result<DetectionIndex> {
	let list = match config.split {
		Voc2007Split::Train => "train",
		Voc2007Split::Validation => "val",
		Voc2007Split::Test => "test",
	};
	let ids = std::fs::read_to_string(root.join("ImageSets/Main").join(format!("{list}.txt")))
		.map_err(|error| Error::io("read voc split list", error))?;
	let mut paths = Vec::new();
	let mut records: Vec<(String, PathBuf)> = Vec::new();
	for id in ids.lines().map(str::trim).filter(|line| !line.is_empty()) {
		let image = root.join("JPEGImages").join(format!("{id}.jpg"));
		let annotation = root.join("Annotations").join(format!("{id}.xml"));
		paths.push(image);
		records.push((id.to_owned(), annotation));
	}
	let mut classes = Vec::new();
	let mut boxes = Vec::new();
	let mut box_lengths = Vec::with_capacity(records.len());
	for (_id, annotation) in &records {
		let text = std::fs::read_to_string(annotation)
			.map_err(|error| Error::io("read voc annotation", error))?;
		let before = boxes.len();
		parse_annotation(&text, &mut classes, &mut boxes)?;
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
