//! CUB-200-2011 bird species as a lazy [`oa::data::Dataset`].
//!
//! `CUB_200_2011.tgz` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `cub200/1.0.0` and its members extract once to
//! `<pack>/extracted/` for reuse. Only the `[N]` F32 label matrix is
//! resident; [`oa::data::Dataset::get_item`] lazily decodes one
//! `[3, H, W]` F32 NCHW image through [`oa::image`], resized to
//! `config.size` when set. Labels are the `NNN.Name` species directory
//! names; the upstream `train_test_split.txt` metadata is not applied —
//! partition with `oa::data::random_split`.

use std::path::Path;

use oa::Result;

use crate::image::folder::ImageIndex;

#[path = "cub200.gen.rs"]
mod generated;

pub(crate) use generated::CUB200_PACK;
pub use generated::{Cub200, Cub200Config};

/// Index `<NNN.Name>/x.jpg` members. The upstream tar scatters class
/// 187's members under a top-level `images/` tree beside
/// `CUB_200_2011/images/`; scanning the extracted root and deriving
/// labels from the parent directory covers both.
fn index(root: &Path, _config: &Cub200Config) -> Result<ImageIndex> {
	ImageIndex::scan(root, |path| {
		let parent = path.parent()?.file_name()?.to_str()?;
		let (number, name) = parent.split_once('.')?;
		if number.len() == 3 && number.bytes().all(|b| b.is_ascii_digit()) && !name.is_empty() {
			Some(parent.to_owned())
		} else {
			None
		}
	})
}
