//! Labeled Faces in the Wild (funneled) as a lazy
//! [`oa::data::Dataset`].
//!
//! `lfw-funneled.tgz` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `lfw/0.4.0` and its `lfw_funneled/<identity>` members extract once
//! to `<pack>/extracted/` for reuse. Only the `[N]` F32 label matrix
//! is resident; [`oa::data::Dataset::get_item`] lazily decodes one
//! `[3, 250, 250]` F32 NCHW image through [`oa::image`], resized to
//! `config.size` when set. Upstream ships no official classification
//! split — partition with `oa::data::random_split`.

#[path = "lfw.gen.rs"]
mod generated;

pub(crate) use generated::LFW_PACK;
pub use generated::{Lfw, LfwConfig};
