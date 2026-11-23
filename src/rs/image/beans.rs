//! Beans leaf disease as a lazy [`oa::data::Dataset`].
//!
//! `train.zip`, `validation.zip`, and `test.zip` are materialized
//! through [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `beans/0.1.0` and their `<split>/<class>` members extract once per
//! split to `<pack>/extracted/` for reuse. Only the `[N]` F32 label
//! matrix is resident; [`oa::data::Dataset::get_item`] lazily decodes
//! one `[3, H, W]` F32 NCHW image through [`oa::image`], resized to
//! `config.size` when set.

#[path = "beans.gen.rs"]
mod generated;

pub(crate) use generated::BEANS_PACK;
pub use generated::{Beans, BeansConfig, BeansSplit};
