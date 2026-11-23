//! Imagenette 320px as a lazy [`oa::data::Dataset`].
//!
//! `imagenette2-320.tgz` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `imagenette/1.0.0` and its `imagenette2-320/{train,val}/<wnid>`
//! members extract once to `<pack>/extracted/` for reuse. Only the
//! `[N]` F32 label matrix is resident;
//! [`oa::data::Dataset::get_item`] lazily decodes one `[3, H, W]` F32
//! NCHW image through [`oa::image`], resized to `config.size` when set.
//! Labels are the ImageNet synset directory names.

#[path = "imagenette.gen.rs"]
mod generated;

pub(crate) use generated::IMAGENETTE_PACK;
pub use generated::{Imagenette, ImagenetteConfig, ImagenetteSplit};
