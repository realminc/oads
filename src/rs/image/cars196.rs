//! Stanford Cars-196 as a lazy [`oa::data::Dataset`].
//!
//! `cars196_{train,test}.zip` are materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `cars196/1.0.0` and extract per split to
//! `<pack>/extracted/<split>/`. Only the `[N]` F32 label matrix is
//! resident; [`oa::data::Dataset::get_item`] lazily decodes one
//! `[3, H, W]` F32 NCHW image through [`oa::image`], resized to
//! `config.size` when set.
//!
//! The mirror ships flat `<split>/NNNNN_<Make>_<Model>_<Year>.jpg`
//! members: the class is the filename stem minus its leading
//! `NNNNN_` ordinal. Stanford's canonical class ordering is
//! alphabetical, which the scan's sorted class table reproduces.

use std::path::Path;

use oa::Result;

use crate::image::folder::ImageIndex;

#[path = "cars196.gen.rs"]
mod generated;

pub(crate) use generated::CARS196_PACK;
pub use generated::{Cars196, Cars196Config, Cars196Split};

/// Index `<split>/NNNNN_<class>.jpg` members: the class is the
/// filename stem minus its leading numeric id.
fn index(root: &Path, _config: &Cars196Config) -> Result<ImageIndex> {
	ImageIndex::scan(root, |path| {
		let stem = path.file_stem()?.to_str()?;
		let (ordinal, class) = stem.split_once('_')?;
		(!ordinal.is_empty() && ordinal.chars().all(|c| c.is_ascii_digit())).then(|| class.to_owned())
	})
}
