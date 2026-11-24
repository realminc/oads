//! MS COCO 2017 object detection as a lazy [`oa::data::Dataset`].
//!
//! `train2017.zip`, `val2017.zip`, and `annotations_trainval2017.zip`
//! extract through [`ensure_pack`](crate::core::pack::ensure_pack)
//! under `coco/1.0.0` per split. `instances_{train,val}2017.json`
//! supplies the image table and object annotations; each `bbox`
//! `[x, y, w, h]` converts to `[x1, y1, x2, y2]` in native pixels and
//! `iscrowd` rows are skipped. Classes follow category-id order (the
//! canonical 80-class table). Only the `[T, 5]` F32 box matrix is
//! resident; [`oa::data::Dataset::get_item`] lazily decodes one
//! `[3, H, W]` image.

use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;

use oa::{Error, Result};
use serde::Deserialize;

use crate::image::folder::DetectionIndex;

#[path = "coco.gen.rs"]
mod generated;

pub(crate) use generated::COCO_PACK;
pub use generated::{Coco, CocoConfig, CocoSplit};

#[derive(Deserialize)]
struct CocoInstances {
	images: Vec<CocoImage>,
	annotations: Vec<CocoAnnotation>,
	categories: Vec<CocoCategory>,
}

#[derive(Deserialize)]
struct CocoImage {
	id: u64,
	file_name: String,
}

#[derive(Deserialize)]
struct CocoAnnotation {
	image_id: u64,
	bbox: [f32; 4],
	category_id: u64,
	iscrowd: u32,
}

#[derive(Deserialize)]
struct CocoCategory {
	id: u64,
	name: String,
}

/// Index `root` (one extracted split tree): `images` rows drive path
/// order under `{split}2017/`; annotations group by `image_id` and
/// remap `category_id` into the id-sorted class table.
fn index(root: &Path, config: &CocoConfig) -> Result<DetectionIndex> {
	let (image_dir, instances) = match config.split {
		CocoSplit::Train => (
			root.join("train2017"),
			root.join("annotations/instances_train2017.json"),
		),
		CocoSplit::Validation => (
			root.join("val2017"),
			root.join("annotations/instances_val2017.json"),
		),
	};
	let file =
		std::fs::File::open(&instances).map_err(|error| Error::io("open coco instances", error))?;
	let instances: CocoInstances = serde_json::from_reader(BufReader::new(file))
		.map_err(|error| Error::data_loss(format!("coco instances json: {error}")))?;
	let mut categories: Vec<_> = instances.categories;
	categories.sort_by_key(|category| category.id);
	let class_of: HashMap<u64, f32> = categories
		.iter()
		.enumerate()
		.map(|(index, category)| (category.id, index as f32))
		.collect();
	let mut boxes: HashMap<u64, Vec<[f32; 5]>> = HashMap::new();
	for annotation in &instances.annotations {
		if annotation.iscrowd != 0 {
			continue;
		}
		let class = class_of
			.get(&annotation.category_id)
			.copied()
			.ok_or_else(|| {
				Error::data_loss(format!(
					"coco annotation category {} is unknown",
					annotation.category_id
				))
			})?;
		let [x, y, w, h] = annotation.bbox;
		boxes
			.entry(annotation.image_id)
			.or_default()
			.push([x, y, x + w, y + h, class]);
	}
	let paths = instances
		.images
		.iter()
		.map(|image| image_dir.join(&image.file_name))
		.collect::<Vec<_>>();
	let mut all_boxes = Vec::new();
	let mut box_lengths = Vec::with_capacity(paths.len());
	for image in &instances.images {
		let before = all_boxes.len();
		if let Some(rows) = boxes.get(&image.id) {
			all_boxes.extend_from_slice(rows);
		}
		box_lengths.push(u32::try_from(all_boxes.len() - before).unwrap_or(u32::MAX));
	}
	Ok(DetectionIndex {
		paths,
		boxes: all_boxes,
		box_lengths,
		classes: categories
			.into_iter()
			.map(|category| category.name)
			.collect(),
	})
}
