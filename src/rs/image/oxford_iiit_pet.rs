//! Oxford-IIIT pet breeds as a lazy [`oa::data::Dataset`].
//!
//! `images.tar.gz` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `oxford_iiit_pet/3.2.0` and its flat `images/<Breed>_<n>.jpg`
//! members extract once to `<pack>/extracted/` for reuse. Only the
//! `[N]` F32 label matrix is resident; [`oa::data::Dataset::get_item`]
//! lazily decodes one `[3, H, W]` F32 NCHW image through
//! [`oa::image`], resized to `config.size` when set. Labels derive from
//! the filename breed stem; the upstream trainval/test lists live in
//! `annotations.tar.gz`, which this pack does not carry.

use std::path::Path;

use oa::Result;

use crate::image::folder::ImageIndex;

#[path = "oxford_iiit_pet.gen.rs"]
mod generated;

pub(crate) use generated::OXFORD_IIIT_PET_PACK;
pub use generated::{OxfordIiitPet, OxfordIiitPetConfig};

/// Index `images/<Breed>_<n>.jpg` members: the class is the filename
/// stem minus its trailing `_<n>` ordinal.
fn index(root: &Path, _config: &OxfordIiitPetConfig) -> Result<ImageIndex> {
	ImageIndex::scan(root, |path| {
		let stem = path.file_stem()?.to_str()?;
		let (breed, ordinal) = stem.rsplit_once('_')?;
		if !ordinal.is_empty() && ordinal.chars().all(|c| c.is_ascii_digit()) {
			Some(breed.to_owned())
		} else {
			None
		}
	})
}
