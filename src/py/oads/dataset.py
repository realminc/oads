"""Shared labeled-image dataset base over device matrices.

Subclasses carry a class-level PACK manifest dict; load() materializes the
pack and decodes it on the host through `oads._native`, then uploads an
[N, C, H, W] f32 image matrix and a [N] f32 label matrix so per-sample CHW
slices collate to NCHW batches through `oa.data`.
"""

from array import array
from pathlib import Path

import oa

from . import _native
from ._progress import resolve


def matrix(field: dict, engine) -> "oa.Matrix":
	"""Upload a native matrix field {"shape": [...], "data": f32 bytes}."""
	values = array("f", field["data"])
	return oa.Matrix.new_from_f32(field["shape"], values, engine)


class ImageDataset:
	"""Device-resident labeled-image split.

	images is [N, C, H, W] f32 and labels is [N] f32 class indices on the
	load engine. get_item returns the CHW sample; get_sample pairs it with
	the [1] label slice.
	"""

	PACK: dict = {}

	@classmethod
	def load(
		cls,
		engine: "oa.Engine | None" = None,
		split: str = "train",
		cache_dir=None,
		normalize: bool = True,
		progress=True,
	) -> "ImageDataset":
		"""Fetch, verify, decode, and upload the selected split.

		cache_dir=None resolves OA_DATA_DIR then the platform cache.
		normalize=True scales pixels into [0, 1]; normalize=False keeps raw
		0..=255 f32 values. progress=True renders the download bar;
		a callable receives (file, fetched, total) samples instead.
		"""
		fields = _native.load(
			cls.PACK["name"],
			split=split,
			normalize=normalize,
			cache_dir=str(Path(cache_dir)) if cache_dir else None,
			progress=resolve(progress),
		)
		return cls(
			matrix(fields["images"], engine),
			matrix(fields["labels"], engine),
			split,
		)

	def __init__(self, images: "oa.Matrix", labels: "oa.Matrix", split: str):
		self._images = images
		self._labels = labels
		self._split = split

	def __len__(self) -> int:
		return self._labels.shape[0]

	@property
	def images(self) -> "oa.Matrix":
		"""Full [N, C, H, W] image matrix on the owning engine."""
		return self._images

	@property
	def labels(self) -> "oa.Matrix":
		"""Full [N] f32 class-index matrix."""
		return self._labels

	@property
	def split(self) -> str:
		return self._split

	def get_item(self, index: int) -> "oa.Matrix":
		"""CHW input slice for one sample index."""
		self._check(index)
		return self._images.slice(0, index, index + 1).reshape(
			list(self._images.shape[1:])
		)

	def get_sample(self, index: int):
		"""(x, y) pair: CHW input slice and [1] label slice."""
		self._check(index)
		return (
			self._images.slice(0, index, index + 1).reshape(
				list(self._images.shape[1:])
			),
			self._labels.slice(0, index, index + 1),
		)

	def _check(self, index: int) -> None:
		if index < 0 or index >= len(self):
			raise IndexError(f"{self.PACK['name']} index is outside the dataset")
