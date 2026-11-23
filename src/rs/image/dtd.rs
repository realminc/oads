//! Describable Textures Dataset as a lazy [`oa::data::Dataset`].
//!
//! `dtd-r1.0.1.tar.gz` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `dtd/3.0.1` and its `dtd/images/<class>/<file>.jpg` members extract
//! once to `<pack>/extracted/` for reuse. Only the `[N]` F32 label
//! matrix is resident; [`oa::data::Dataset::get_item`] lazily decodes
//! one `[3, H, W]` F32 NCHW image through [`oa::image`], resized to
//! `config.size` when set. Splits follow the upstream fold-1 lists —
//! `labels/{train,val,test}1.txt` name `<class>/<file>.jpg` members.

use std::collections::HashSet;
use std::path::Path;

use oa::{Error, Result};

use crate::image::folder::ImageIndex;

#[path = "dtd.gen.rs"]
mod generated;

pub(crate) use generated::DTD_PACK;
pub use generated::{Dtd, DtdConfig, DtdSplit};

/// Index `images/<class>/<file>.jpg` members assigned to
/// `config.split` by the fold-1 `labels/{train,val,test}1.txt` lists.
fn index(root: &Path, config: &DtdConfig) -> Result<ImageIndex> {
	let tag = match config.split {
		DtdSplit::Train => "train1",
		DtdSplit::Validation => "val1",
		DtdSplit::Test => "test1",
	};
	let manifest = std::fs::read_to_string(root.join("labels").join(format!("{tag}.txt")))
		.map_err(|error| Error::io("read dtd fold manifest", error))?;
	let members: HashSet<&str> = manifest.lines().collect();
	ImageIndex::scan(&root.join("images"), |path| {
		let class = path.parent()?.file_name()?.to_str()?;
		let file = path.file_name()?.to_str()?;
		let key = format!("{class}/{file}");
		members.contains(key.as_str()).then(|| class.to_owned())
	})
}
