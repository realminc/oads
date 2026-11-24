//! KITTI 2D object detection as a lazy [`oa::data::Dataset`].
//!
//! `data_object_image_2.zip` and `data_object_label_2.zip` extract
//! through [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `kitti/1.0.0` per split. Training ids carry one `label_2/{id}.txt`
//! per image: whitespace-separated records whose fields 4-7 are
//! `[x1, y1, x2, y2]` in pixels and whose leading token is the class
//! name. `DontCare` rows are not objects and are skipped. Testing ids
//! ship no labels, so test samples slice to `[0, 5]` boxes. Only the
//! `[T, 5]` F32 box matrix is resident;
//! [`oa::data::Dataset::get_item`] lazily decodes one `[3, H, W]`
//! image.

use std::path::Path;

use oa::{Error, Result};

use crate::image::folder::DetectionIndex;

#[path = "kitti.gen.rs"]
mod generated;

pub(crate) use generated::KITTI_PACK;
pub use generated::{Kitti, KittiConfig, KittiSplit};

/// Index `root` (one extracted split tree): `image_2/{id}.png` members
/// drive path order; `label_2/{id}.txt` contributes boxes where it
/// exists. Classes sort alphabetically.
fn index(root: &Path, config: &KittiConfig) -> Result<DetectionIndex> {
	let image_dir = match config.split {
		KittiSplit::Train => root.join("training/image_2"),
		KittiSplit::Test => root.join("testing/image_2"),
	};
	let mut paths: Vec<_> = std::fs::read_dir(&image_dir)
		.map_err(|error| Error::io("list kitti images", error))?
		.filter_map(|entry| entry.ok().map(|entry| entry.path()))
		.filter(|path| path.extension().is_some_and(|ext| ext == "png"))
		.collect();
	paths.sort();
	let label_dir = root.join("training/label_2");
	let mut classes = Vec::new();
	let mut boxes = Vec::new();
	let mut box_lengths = Vec::with_capacity(paths.len());
	for image in &paths {
		let before = boxes.len();
		if config.split == KittiSplit::Train {
			let id = image
				.file_stem()
				.ok_or_else(|| Error::data_loss("kitti image member has no stem"))?;
			let text = std::fs::read_to_string(label_dir.join(format!("{}.txt", id.to_string_lossy())))
				.map_err(|error| Error::io("read kitti label", error))?;
			parse_labels(&text, &mut classes, &mut boxes)?;
		}
		box_lengths.push(u32::try_from(boxes.len() - before).unwrap_or(u32::MAX));
	}
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

/// One KITTI label line is
/// `type truncated occluded alpha x1 y1 x2 y2 dims(3) loc(3) rot_y`;
/// fields 4-8 are the box. `DontCare` rows are skipped.
fn parse_labels(text: &str, classes: &mut Vec<String>, out: &mut Vec<[f32; 5]>) -> Result<()> {
	for line in text.lines() {
		let mut fields = line.split_whitespace();
		let name = fields
			.next()
			.ok_or_else(|| Error::data_loss("kitti label line is empty"))?;
		if name == "DontCare" {
			continue;
		}
		let field = |index: usize, line: &str| -> Result<f32> {
			line
				.split_whitespace()
				.nth(index)
				.ok_or_else(|| Error::data_loss("kitti label line is truncated"))?
				.parse()
				.map_err(|_| Error::data_loss("kitti label field is not a number"))
		};
		let class = match classes.iter().position(|class| class == name) {
			Some(index) => index,
			None => {
				classes.push(name.to_owned());
				classes.len() - 1
			}
		};
		out.push([
			field(4, line)?,
			field(5, line)?,
			field(6, line)?,
			field(7, line)?,
			class as f32,
		]);
	}
	Ok(())
}
