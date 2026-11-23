//! Imagewang 320px as a lazy [`oa::data::Dataset`].
//!
//! `imagewang-320.tgz` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `imagewang/1.0.0` and its `imagewang-320/{train,val}/<wnid>`
//! members extract once to `<pack>/extracted/` for reuse — the
//! unlabeled `unsup/` tree is fetched with the archive but not
//! indexed. Only the `[N]` F32 label matrix is resident;
//! [`oa::data::Dataset::get_item`] lazily decodes one `[3, H, W]` F32
//! NCHW image through [`oa::image`], resized to `config.size` when set.

#[path = "imagewang.gen.rs"]
mod generated;

pub(crate) use generated::IMAGEWANG_PACK;
pub use generated::{Imagewang, ImagewangConfig, ImagewangSplit};
