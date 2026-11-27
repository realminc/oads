"""Imagenette 320px — 9,469 train + 3,925 validation JPEGs over
10 ImageNet synset directories.
"""

from array import array
from pathlib import Path

import oa

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("imagenette")


class Imagenette:
	"""Lazy [1, 3, H, W] f32 NCHW images + [N] class labels."""

	def __init__(self, labels, label_list, paths, classes, size, engine):
		self._labels = labels
		self._label_list = label_list
		self._paths = paths
		self._classes = classes
		self._size = size
		self._engine = engine

	@classmethod
	def load(
		cls,
		engine: "oa.Runtime | None" = None,
		split: str = "train",
		size=None,
		cache_dir=None,
		progress=True,
	) -> "Imagenette":
		"""Fetch, verify, extract, and index the pack.

		Members stay in the extracted cache and decode lazily per
		sample. `size=(width, height)` resizes each image on access;
		progress=True renders the download bar; a callable receives
		(file, fetched, total) samples instead.
		"""
		fields = _native.load(
			PACK["name"],
			split=split,
			cache_dir=str(Path(cache_dir)) if cache_dir else None,
			progress=resolve(progress),
		)
		resolved = engine if engine is not None else oa.Runtime.current()
		root = Path(cache_dir) if cache_dir else Path(_native.data_root())
		extracted = root / PACK["name"] / PACK["version"] / "extracted"
		paths = [extracted / line for line in fields["text"].splitlines()]
		classes = (extracted / f".classes_{split}").read_text().splitlines()
		label_list = [int(v) for v in array("f", fields["labels"]["data"])]
		return cls(
			matrix(fields["labels"], resolved),
			label_list,
			paths,
			classes,
			size,
			resolved,
		)

	def __len__(self) -> int:
		return len(self._paths)

	@property
	def labels(self) -> "oa.Matrix":
		"""[N] f32 class indices into `classes`."""
		return self._labels

	@property
	def classes(self) -> list:
		"""Class names in label-index order."""
		return self._classes

	def path(self, index: int) -> Path:
		"""Extracted image path for row `index`."""
		return self._paths[index]

	def label(self, index: int) -> int:
		"""Class index for row `index` (host-side, no readback)."""
		return self._label_list[index]

	def image(self, index: int) -> "oa.Image":
		"""Lazily decode image `index` to a [1, 3, H, W] f32 NCHW image."""
		image = oa.image.decode_file(
			str(self._paths[index]), "rgb", self._engine
		)
		if self._size is not None:
			width, height = self._size
			return oa.image.resize(image, width, height)
		return image

	def get_item(self, index: int) -> "oa.Matrix":
		"""Lazily decode image `index` as a [3, H, W] matrix."""
		item = self.image(index).as_matrix()
		return item.reshape(item.shape[1:])

	def get_sample(self, index: int):
		"""(image, class label) pair for one row."""
		return self.get_item(index), self._labels.slice(0, index, index + 1)
