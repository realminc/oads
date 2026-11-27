"""Moving MNIST frame sequences as a device-resident dataset.

mnist_test_seq.npy is materialized through `oads.pack` under
moving_mnist/1.0.0 and uploaded once as a [N*T, 1, 64, 64] f32 matrix:
10,000 sequences of 20 grayscale frames laid out sequence-major. A sample is
one whole sequence [T, 1, 64, 64]; get_sample pairs frames[0..T-1] with the
one-frame-ahead frames[1..T] — the next-frame-prediction contract. Storage
stays rank-4 because `oa.matrix.slice` is rank-one through rank-four.
"""

import oa

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("moving_mnist")


class MovingMnist:
	def __init__(self, frames: "oa.Matrix", seq_len: int):
		self._frames = frames
		self._seq_len = seq_len

	@classmethod
	def load(
		cls,
		engine: "oa.Runtime | None" = None,
		count=None,
		normalize: bool = True,
		cache_dir=None,
		progress=True,
	) -> "MovingMnist":
		"""Fetch, verify, decode, and upload up to `count` sequences.

		count=None loads all 10,000 — roughly 3.3 GiB of f32 device memory.
		progress=True renders the download bar; a callable receives
		(file, fetched, total) samples instead.
		"""
		fields = _native.load(
			PACK["name"],
			normalize=normalize,
			count=count,
			cache_dir=str(cache_dir) if cache_dir else None,
			progress=resolve(progress),
		)
		return cls(matrix(fields["frames"], engine), fields["seq_len"])

	def __len__(self) -> int:
		return self._frames.shape[0] // self._seq_len

	@property
	def frames(self) -> "oa.Matrix":
		"""Full [N*T, 1, 64, 64] f32 frame matrix on the owning engine."""
		return self._frames

	@property
	def seq_len(self) -> int:
		return self._seq_len

	def get_item(self, index: int) -> "oa.Matrix":
		"""[T, 1, 64, 64] frame sequence for one sample index."""
		self._check(index)
		return self._frames.slice(
			0, index * self._seq_len, (index + 1) * self._seq_len
		)

	def get_sample(self, index: int):
		"""(x, y): [T-1, 1, 64, 64] frames and the one-ahead target."""
		self._check(index)
		start = index * self._seq_len
		end = start + self._seq_len
		return (
			self._frames.slice(0, start, end - 1),
			self._frames.slice(0, start + 1, end),
		)

	def _check(self, index: int) -> None:
		if index < 0 or index >= len(self):
			raise IndexError("moving_mnist index is outside the dataset")
