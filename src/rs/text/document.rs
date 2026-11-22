//! Shared byte-window encoding for text-classification loaders.
//!
//! Each document is emitted as `text_len` F32 values of its UTF-8 bytes,
//! truncated or zero-padded; `normalize` scales by `1/255` into `[0, 1]`
//! like the image loaders. Labels ride as `[N]` F32 class indices, so the
//! pair uploads to [`oa::data::MatrixDataset`] like any fixed-shape pack.

use oa::Result;

use crate::decode::MatrixData;

/// Encode `(label, utf-8 text)` pairs as a `[N, text_len]` F32 window
/// matrix and a `[N]` F32 label matrix.
///
/// # Errors
///
/// Infallible today; returns `Result` so callers share the decode signature.
pub(crate) fn encode(
	field: &'static str,
	documents: &[(f32, Vec<u8>)],
	text_len: usize,
	normalize: bool,
) -> Result<(MatrixData, MatrixData)> {
	let scale = if normalize { 1.0 / 255.0 } else { 1.0 };
	let mut values = vec![0.0_f32; documents.len() * text_len];
	for (row, (_, text)) in documents.iter().enumerate() {
		let take = text.len().min(text_len);
		let row_out = &mut values[row * text_len..row * text_len + take];
		for (out, &byte) in row_out.iter_mut().zip(&text[..take]) {
			*out = f32::from(byte) * scale;
		}
	}
	let text = MatrixData {
		name: field,
		shape: vec![documents.len(), text_len],
		values,
	};
	let labels = MatrixData {
		name: "labels",
		shape: vec![documents.len()],
		values: documents.iter().map(|&(label, _)| label).collect(),
	};
	Ok((text, labels))
}
