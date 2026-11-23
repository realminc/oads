//! Rock-paper-scissors hand gestures as a lazy [`oa::data::Dataset`].
//!
//! `rps.zip` and `rps-test-set.zip` are materialized through
//! [`ensure_pack`](crate::core::pack::ensure_pack) under
//! `rock_paper_scissors/1.0.0` and their `rps/{paper,rock,scissors}` /
//! `rps-test-set/{...}` members extract once to `<pack>/extracted/` for
//! reuse. Only the `[N]` F32 label matrix is resident;
//! [`oa::data::Dataset::get_item`] lazily decodes one `[3, H, W]` F32
//! NCHW image through [`oa::image`], resized to `config.size` when set.

#[path = "rock_paper_scissors.gen.rs"]
mod generated;

pub(crate) use generated::ROCK_PAPER_SCISSORS_PACK;
pub use generated::{RockPaperScissors, RockPaperScissorsConfig, RockPaperScissorsSplit};
