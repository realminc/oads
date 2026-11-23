//! MIT CSAIL Places365-standard small as a lazy
//! [`oa::data::Dataset`].
//!
//! `places365standard_easyformat.tar` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `places365_small/1.0.0` and its
//! `places365_standard/{train,val}/<class>/` members extract once to
//! `<pack>/extracted/` for reuse — roughly 1.8M files, so both
//! indexing and decode stay lazy. Only the `[N]` F32 label matrix is
//! resident; [`oa::data::Dataset::get_item`] lazily decodes one
//! `[3, 256, 256]` F32 NCHW image through [`oa::image`], resized to
//! `config.size` when set.

#[path = "places365_small.gen.rs"]
mod generated;

pub(crate) use generated::PLACES365_SMALL_PACK;
pub use generated::{Places365Small, Places365SmallConfig, Places365SmallSplit};
