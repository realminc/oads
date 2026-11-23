//! Omniglot handwritten-character corpus as a lazy imagefolder
//! [`oa::data::Dataset`].
//!
//! `images_background.zip` and `images_evaluation.zip` are materialized
//! through [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `omniglot/1.0.0` and extracted per split. Each
//! `<Alphabet>/<character>/` member becomes one class, so labels index
//! the 964-way background or 659-way evaluation class tables; the
//! alphabet name is the `Alphabet/` prefix inside each class name.

use std::path::Path;

use oa::Result;

use crate::image::folder::ImageIndex;

#[path = "omniglot.gen.rs"]
mod generated;

pub(crate) use generated::OMNIGLOT_PACK;
pub use generated::{Omniglot, OmniglotConfig, OmniglotSplit};

/// Index `<root>/images_{background,evaluation}/<Alphabet>/<char>/*.png`
/// members: the class is the `<Alphabet>/<char>` pair, since character
/// directory names repeat across alphabets.
fn index(root: &Path, _config: &OmniglotConfig) -> Result<ImageIndex> {
	ImageIndex::scan(root, |path| {
		let character = path.parent()?;
		let alphabet = character.parent()?;
		Some(format!(
			"{}/{}",
			alphabet.file_name()?.to_str()?,
			character.file_name()?.to_str()?
		))
	})
}
