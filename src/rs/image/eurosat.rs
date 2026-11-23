//! EuroSAT Sentinel-2 land use as a lazy [`oa::data::Dataset`].
//!
//! `EuroSAT.zip` is materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `eurosat/2.0.0` and its `2750/<Class>` members extract once to
//! `<pack>/extracted/` for reuse. Only the `[N]` F32 label matrix is
//! resident; [`oa::data::Dataset::get_item`] lazily decodes one
//! `[3, 64, 64]` F32 NCHW image through [`oa::image`], resized to
//! `config.size` when set. Upstream ships no official split —
//! partition with `oa::data::random_split`.

#[path = "eurosat.gen.rs"]
mod generated;

pub(crate) use generated::EUROSAT_PACK;
pub use generated::{Eurosat, EurosatConfig};
