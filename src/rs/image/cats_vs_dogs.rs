//! Microsoft Kaggle cats vs dogs as a lazy [`oa::data::Dataset`].
//!
//! `kagglecatsanddogs_5340.zip` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under `cats_vs_dogs/1.0.0`
//! and its `PetImages/{Cat,Dog}` members extract once to
//! `<pack>/extracted/` for reuse. Only the `[N]` F32 label matrix is
//! resident; [`oa::data::Dataset::get_item`] lazily decodes one
//! `[1, 3, H, W]` F32 NCHW image through [`oa::image`], resized to
//! `CatsVsDogsConfig.size` when set.
//!
//! A few upstream members are corrupt JPEGs (a documented Kaggle issue);
//! the index includes them and decode errors surface lazily on access.
//! The corpus ships no official split — partition with
//! [`oa::data::random_split`], or set a uniform `size` and collate.

#[path = "cats_vs_dogs.gen.rs"]
mod generated;

pub(crate) use generated::CATS_VS_DOGS_PACK;
pub use generated::{CLASSES, CatsVsDogs, CatsVsDogsConfig};
