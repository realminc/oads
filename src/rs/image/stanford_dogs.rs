//! Stanford dogs as a lazy [`oa::data::Dataset`].
//!
//! `images.tar` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `stanford_dogs/0.2.0` and its `Images/n<wnid>-<breed>` members
//! extract once to `<pack>/extracted/` for reuse. Only the `[N]` F32
//! label matrix is resident; [`oa::data::Dataset::get_item`] lazily
//! decodes one `[3, H, W]` F32 NCHW image through [`oa::image`],
//! resized to `config.size` when set. Labels are the synset directory
//! names; the upstream split lives in the separate annotation tar,
//! which this pack does not carry.

#[path = "stanford_dogs.gen.rs"]
mod generated;

pub(crate) use generated::STANFORD_DOGS_PACK;
pub use generated::{StanfordDogs, StanfordDogsConfig};
