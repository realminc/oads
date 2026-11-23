//! Internal pack mechanics shared by the typed loaders.
//!
//! `core` is the machinery layer beneath the domain modules: [`pack`] is
//! the transport boundary (manifests, `OA_DATA_DIR` cache resolution,
//! resumable fetch, SHAKE256-256 verification), [`decode`] is the host-side
//! decode boundary shared by device loaders and foreign bindings, and the
//! remaining modules are narrow format readers — CSV, archive extraction,
//! FLAC STREAMINFO, IDX, JSONL, MAT 5.0, MIDI/SMF, NPY, and WAV — used only
//! to materialize upstream payloads.
//!
//! `pack` and `decode` are public API and re-exported at the crate root;
//! everything else is `pub(crate)` implementation detail and must not grow
//! into a stable surface.

pub(crate) mod csv;
pub mod decode;
#[path = "core/decode_dispatch.gen.rs"]
mod decode_dispatch;
pub(crate) mod extract;
pub(crate) mod flac;
pub(crate) mod idx;
pub(crate) mod jsonl;
pub(crate) mod mat;
pub(crate) mod midi;
pub(crate) mod npy;
pub mod pack;
pub(crate) mod tsv;
pub(crate) mod wav;

pub use idx::IdxSplit;
