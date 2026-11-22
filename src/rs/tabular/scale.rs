//! Per-column scaling for tabular feature matrices.

/// Scale each column of a row-major `[N, cols]` F32 buffer to `[0, 1]` by its
/// min-max range; a constant column becomes all zeros.
pub(crate) fn normalize_columns(values: &mut [f32], cols: usize) {
	for col in 0..cols {
		let mut min = f32::INFINITY;
		let mut max = f32::NEG_INFINITY;
		for row in values.chunks_exact(cols) {
			min = min.min(row[col]);
			max = max.max(row[col]);
		}
		let range = max - min;
		for row in values.chunks_exact_mut(cols) {
			row[col] = if range > 0.0 {
				(row[col] - min) / range
			} else {
				0.0
			};
		}
	}
}
