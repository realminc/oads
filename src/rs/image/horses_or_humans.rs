//! Horses-or-humans as a lazy [`oa::data::Dataset`].
//!
//! `horse-or-human.zip` and `validation-horse-or-human.zip` are
//! materialized through [`ensure_pack`](crate::core::pack::ensure_pack)
//! under `horses_or_humans/1.0.0` and their `{horses,humans}` members
//! extract once per split to `<pack>/extracted/` for reuse. Only the
//! `[N]` F32 label matrix is resident;
//! [`oa::data::Dataset::get_item`] lazily decodes one `[3, H, W]` F32
//! NCHW image through [`oa::image`], resized to `config.size` when set.

#[path = "horses_or_humans.gen.rs"]
mod generated;

pub(crate) use generated::HORSES_OR_HUMANS_PACK;
pub use generated::{HorsesOrHumans, HorsesOrHumansConfig, HorsesOrHumansSplit};
