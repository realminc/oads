//! UC Merced land use as a lazy [`oa::data::Dataset`].
//!
//! `UCMerced_LandUse.zip` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `uc_merced/1.0.0` and its `UCMerced_LandUse/Images/<class>` TIFF
//! members extract once to `<pack>/extracted/` for reuse. Only the
//! `[N]` F32 label matrix is resident;
//! [`oa::data::Dataset::get_item`] lazily decodes one
//! `[3, 256, 256]` F32 NCHW image through [`oa::image`], resized to
//! `config.size` when set. Upstream ships no official split —
//! partition with `oa::data::random_split`.

#[path = "uc_merced.gen.rs"]
mod generated;

pub(crate) use generated::UC_MERCED_PACK;
pub use generated::{UcMerced, UcMercedConfig};
