//! Oxford 102-category flowers as a lazy [`oa::data::Dataset`].
//!
//! `102flowers.tgz` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `oxford_flowers102/2.0.0` and its flat `jpg/image_NNNNN.jpg` members
//! extract once to `<pack>/extracted/` for reuse. Only the `[N]` F32
//! label matrix is resident; [`oa::data::Dataset::get_item`] lazily
//! decodes one `[3, H, W]` F32 NCHW image through [`oa::image`],
//! resized to `config.size` when set. Class names follow the published
//! Oxford VGG ordering; label `k` indexes `NAMES[k]`.
//!
//! The official train/validation/test assignment ships outside the
//! image archive: `setid.mat` carries the `trnid`/`valid`/`tstid` id
//! lists (1,020/1,020/6,149 of the 8,189 images) and `imagelabels.mat`
//! carries one 1-based class id per image. Both are MAT 5.0 files
//! parsed through the core `mat` reader at index time.

use std::collections::HashSet;
use std::path::Path;

use oa::{Error, Result};

use crate::core::mat;
use crate::image::folder::{ImageIndex, collect_images};

#[path = "oxford_flowers102.gen.rs"]
mod generated;

pub(crate) use generated::OXFORD_FLOWERS102_PACK;
pub use generated::{OxfordFlowers102, OxfordFlowers102Config, OxfordFlowers102Split};

/// Class names in label order (Oxford VGG published ordering, shared
/// by TFDS `oxford_flowers102`).
pub const NAMES: [&str; 102] = [
	"pink primrose",
	"hard-leaved pocket orchid",
	"canterbury bells",
	"sweet pea",
	"english marigold",
	"tiger lily",
	"moon orchid",
	"bird of paradise",
	"monkshood",
	"globe thistle",
	"snapdragon",
	"colt's foot",
	"king protea",
	"spear thistle",
	"yellow iris",
	"globe-flower",
	"purple coneflower",
	"peruvian lily",
	"balloon flower",
	"giant white arum lily",
	"fire lily",
	"pincushion flower",
	"fritillary",
	"red ginger",
	"grape hyacinth",
	"corn poppy",
	"prince of wales feathers",
	"stemless gentian",
	"artichoke",
	"sweet william",
	"carnation",
	"garden phlox",
	"love in the mist",
	"mexican aster",
	"alpine sea holly",
	"ruby-lipped cattleya",
	"cape flower",
	"great masterwort",
	"siam tulip",
	"lenten rose",
	"barbeton daisy",
	"daffodil",
	"sword lily",
	"poinsettia",
	"bolero deep blue",
	"wallflower",
	"marigold",
	"buttercup",
	"oxeye daisy",
	"common dandelion",
	"petunia",
	"wild pansy",
	"primula",
	"sunflower",
	"pelargonium",
	"bishop of llandaff",
	"gaura",
	"geranium",
	"orange dahlia",
	"pink-yellow dahlia?",
	"cautleya spicata",
	"japanese anemone",
	"black-eyed susan",
	"silverbush",
	"californian poppy",
	"osteospermum",
	"spring crocus",
	"bearded iris",
	"windflower",
	"tree poppy",
	"gazania",
	"azalea",
	"water lily",
	"rose",
	"thorn apple",
	"morning glory",
	"passion flower",
	"lotus",
	"toad lily",
	"anthurium",
	"frangipani",
	"clematis",
	"hibiscus",
	"columbine",
	"desert-rose",
	"tree mallow",
	"magnolia",
	"cyclamen",
	"watercress",
	"canna lily",
	"hippeastrum",
	"bee balm",
	"ball moss",
	"foxglove",
	"bougainvillea",
	"camellia",
	"mallow",
	"mexican petunia",
	"bromelia",
	"blanket flower",
	"trumpet creeper",
	"blackberry lily",
];

/// Index `jpg/image_NNNNN.jpg` members assigned to `config.split` by
/// `setid.mat`, labeled by `imagelabels.mat`.
fn index(root: &Path, config: &OxfordFlowers102Config) -> Result<ImageIndex> {
	// `root` is `<pack>/extracted/jpg`; the .mat sidecars sit in the pack
	// dir fetched by `ensure_pack`.
	let pack_dir = root
		.parent()
		.and_then(|extracted| extracted.parent())
		.ok_or_else(|| Error::internal("oxford_flowers102 index root has no pack dir"))?;
	let labels_bytes = std::fs::read(pack_dir.join("imagelabels.mat"))
		.map_err(|error| Error::io("read oxford_flowers102 imagelabels.mat", error))?;
	let setid_bytes = std::fs::read(pack_dir.join("setid.mat"))
		.map_err(|error| Error::io("read oxford_flowers102 setid.mat", error))?;
	let labels_map = mat::read_arrays("oxford_flowers102", &labels_bytes)?;
	let image_labels = labels_map
		.get("labels")
		.ok_or_else(|| Error::data_loss("oxford_flowers102 imagelabels.mat lacks a labels array"))?;
	if image_labels.dims.iter().product::<usize>() != image_labels.values.len()
		|| image_labels.values.len() != 8_189
	{
		return Err(Error::data_loss(
			"oxford_flowers102 imagelabels.mat does not hold 8,189 labels",
		));
	}
	let set_map = mat::read_arrays("oxford_flowers102", &setid_bytes)?;
	let key = match config.split {
		OxfordFlowers102Split::Train => "trnid",
		OxfordFlowers102Split::Validation => "valid",
		OxfordFlowers102Split::Test => "tstid",
	};
	let ids = set_map
		.get(key)
		.ok_or_else(|| Error::data_loss(format!("oxford_flowers102 setid.mat lacks a {key} array")))?;
	let wanted: HashSet<u32> = ids.values.iter().map(|&id| id as u32).collect();

	let mut files = Vec::new();
	collect_images(root, &mut files)?;
	files.sort();
	let mut paths = Vec::new();
	let mut labels = Vec::new();
	for path in files {
		let stem = match path.file_stem().and_then(|s| s.to_str()) {
			Some(stem) => stem,
			None => continue,
		};
		let id: u32 = match stem.strip_prefix("image_").and_then(|s| s.parse().ok()) {
			Some(id) => id,
			None => continue,
		};
		if !wanted.contains(&id) {
			continue;
		}
		let row = usize::try_from(id - 1)
			.map_err(|_| Error::data_loss("oxford_flowers102 image id is out of range"))?;
		let label = image_labels
			.values
			.get(row)
			.ok_or_else(|| Error::data_loss("oxford_flowers102 imagelabels.mat is short a row"))?;
		let label = u16::try_from(*label as i64 - 1)
			.map_err(|_| Error::data_loss("oxford_flowers102 label escapes the u16 encoding"))?;
		paths.push(path);
		labels.push(label);
	}
	Ok(ImageIndex {
		paths,
		labels,
		classes: NAMES.iter().map(|name| (*name).to_owned()).collect(),
	})
}
