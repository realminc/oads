//! Host-side pack decode shared by device loaders and foreign bindings.
//!
//! Every loader's `decode` materializes its pack through
//! [`ensure_pack`](crate::pack::ensure_pack), decodes members into
//! [`MatrixData`] F32 buffers, and returns a [`Decoded`] bundle. Rust
//! loaders upload the matrices to an [`oa::Engine`]; the Python extension
//! returns the same buffers for `oa.Matrix` construction there, so decode
//! logic exists exactly once.
//!
//! Upload produces F32 everywhere — `oa::matrix::slice` is F32-only — so
//! `normalize` selects a `[0, 1]` scale or raw `0..=255` F32 values, never
//! a different dtype.

use std::path::PathBuf;

use oa::matrix::Matrix;
use oa::{Engine, Error, Result};

use crate::idx::IdxSplit;
use crate::pack::{FetchProgress, Pack};

/// One decoded matrix: a shape and a row-major F32 payload whose product
/// matches the shape.
pub struct MatrixData {
	/// Field name loaders and bindings agree on: `images`, `labels`,
	/// `tokens`, or `frames`.
	pub name: &'static str,
	/// Tensor shape in OA order (`[N, C, H, W]`, `[N]`, `[N*T, 1, H, W]`).
	pub shape: Vec<usize>,
	/// Row-major F32 payload in native byte order.
	pub values: Vec<f32>,
}

/// A pack decoded on the host: named matrices plus optional corpus metadata.
pub struct Decoded {
	/// Named matrices in loader-defined order.
	pub matrices: Vec<MatrixData>,
	/// Decoded corpus text when the pack is a text corpus.
	pub text: Option<String>,
	/// Sequence length when the pack materializes frame sequences.
	pub seq_len: usize,
}

impl Decoded {
	/// Borrow a named matrix.
	///
	/// # Errors
	///
	/// Returns an error when the decode produced no matrix under `name`.
	pub fn matrix(&self, name: &str) -> Result<&MatrixData> {
		self
			.matrices
			.iter()
			.find(|matrix| matrix.name == name)
			.ok_or_else(|| Error::data_loss(format!("decoded pack has no '{name}' matrix")))
	}

	/// Upload a named matrix to `engine` as an F32 [`Matrix`].
	///
	/// # Errors
	///
	/// Returns an error for a missing matrix name or a failed upload.
	pub fn upload(&self, name: &str, engine: &Engine) -> Result<Matrix> {
		let matrix = self.matrix(name)?;
		Matrix::new_from_slice_on(engine, matrix.shape.clone(), &matrix.values)
	}
}

/// Load-time options accepted by [`load`]; loaders ignore fields that do
/// not apply to their pack.
#[derive(Clone, Debug)]
pub struct DecodeConfig {
	/// Train/test split for split packs.
	pub split: IdxSplit,
	/// Scale image pixels into `[0, 1]`; `false` keeps raw `0..=255` F32.
	pub normalize: bool,
	/// Bound materialized sample count where the pack supports it
	/// (currently `moving_mnist`); `None` decodes the full pack.
	pub count: Option<usize>,
	/// Cache root override; `None` resolves `OA_DATA_DIR`, then the
	/// platform cache.
	pub cache_dir: Option<PathBuf>,
}

impl Default for DecodeConfig {
	fn default() -> Self {
		Self {
			split: IdxSplit::Train,
			normalize: true,
			count: None,
			cache_dir: None,
		}
	}
}

/// Return the compiled manifest for a registered dataset pack.
///
/// # Errors
///
/// Returns an error when `name` is not a registered pack.
pub fn manifest(name: &str) -> Result<&'static Pack> {
	crate::decode_dispatch::manifest_of(name)
}

/// Materialize and decode a registered pack on the host.
///
/// Dispatches to the owning loader's `decode` with the matching typed
/// config; `DecodeConfig` fields a loader does not use are ignored.
///
/// # Errors
///
/// Returns an error for an unknown pack name, pack fetch/verification
/// failures, and member decode failures.
pub fn load(name: &str, config: &DecodeConfig) -> Result<Decoded> {
	load_with_progress(name, config, &mut |_| true)
}

/// [`load`] with a fetch progress callback; see
/// [`ensure_pack_with_progress`](crate::pack::ensure_pack_with_progress).
///
/// # Errors
///
/// Returns the errors of [`load`], plus `Cancelled` when the callback
/// aborts a fetch.
pub fn load_with_progress(
	name: &str,
	config: &DecodeConfig,
	progress: &mut impl FnMut(FetchProgress) -> bool,
) -> Result<Decoded> {
	crate::decode_dispatch::load_of(name, config, progress)
}
