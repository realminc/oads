//! Food-101 dishes as a lazy [`oa::data::Dataset`].
//!
//! `food-101.tar.gz` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `food101/3.0.0` and its `food-101/images/<class>/<id>.jpg` members
//! extract once to `<pack>/extracted/` for reuse. Only the `[N]` F32
//! label matrix is resident; [`oa::data::Dataset::get_item`] lazily
//! decodes one `[3, H, W]` F32 NCHW image through [`oa::image`],
//! resized to `config.size` when set. The official train/test
//! assignment comes from `food-101/meta/{train,test}.txt`, which lists
//! `<class>/<id>` members without extensions.

use std::collections::HashSet;
use std::path::Path;

use oa::{Error, Result};

use crate::image::folder::ImageIndex;

#[path = "food101.gen.rs"]
mod generated;

pub(crate) use generated::FOOD101_PACK;
pub use generated::{Food101, Food101Config, Food101Split};

/// Index `images/<class>/<id>.jpg` members whose `<class>/<id>` pair is
/// assigned to `config.split` by `meta/{train,test}.txt`.
fn index(root: &Path, config: &Food101Config) -> Result<ImageIndex> {
	let tag = match config.split {
		Food101Split::Train => "train",
		Food101Split::Test => "test",
	};
	let meta = std::fs::read_to_string(root.join("meta").join(format!("{tag}.txt")))
		.map_err(|error| Error::io("read food101 split manifest", error))?;
	let members: HashSet<&str> = meta.lines().collect();
	ImageIndex::scan(&root.join("images"), |path| {
		let class = path.parent()?.file_name()?.to_str()?;
		let id = path.file_stem()?.to_str()?;
		let key = format!("{class}/{id}");
		members.contains(key.as_str()).then(|| class.to_owned())
	})
}
