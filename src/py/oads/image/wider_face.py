"""WIDER FACE detection — single `face` class, `invalid` ground
truth rows skipped; lazy images plus `[x1,y1,x2,y2,class]` boxes.
Test ships a filelist only.
"""

from array import array
from itertools import accumulate
from pathlib import Path

import oa

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("wider_face")


class WiderFace:
	"""Lazy [3, H, W] f32 NCHW images + [K, 5] [x1,y1,x2,y2,class] boxes."""

	def __init__(self, boxes, lengths, offsets, paths, classes, engine):
		self._boxes = boxes
		self._lengths = lengths
		self._offsets = offsets
		self._paths = paths
		self._classes = classes
		self._engine = engine

	@classmethod
	def load(
		cls,
		engine: "oa.Engine | None" = None,
		split: str = "train",
		cache_dir=None,
		progress=True,
	) -> "WiderFace":
		"""Fetch, verify, extract, and index the pack.

		Members stay in the extracted cache and decode lazily per
		sample. `progress=True` renders the download bar; a callable
		receives (file, fetched, total) samples instead.
		"""
		fields = _native.load(
			PACK["name"],
			split=split,
			cache_dir=str(Path(cache_dir)) if cache_dir else None,
			progress=resolve(progress),
		)
		resolved = engine if engine is not None else oa.Engine.current()
		root = Path(cache_dir) if cache_dir else Path(_native.data_root())
		extracted = root / PACK["name"] / PACK["version"] / "extracted"
		paths = [extracted / line for line in fields["text"].splitlines()]
		classes = (extracted / f".classes_{split}").read_text().splitlines()
		lengths = [int(v) for v in array("f", fields["box_lengths"]["data"])]
		offsets = [0, *accumulate(lengths)]
		return cls(
			matrix(fields["boxes"], resolved),
			lengths,
			offsets,
			paths,
			classes,
			resolved,
		)

	def __len__(self) -> int:
		return len(self._paths)

	@property
	def boxes(self) -> "oa.Matrix":
		"""[T, 5] f32 [x1,y1,x2,y2,class] boxes in native pixels."""
		return self._boxes

	@property
	def classes(self) -> list:
		"""Class names in box-label order."""
		return self._classes

	def path(self, index: int) -> Path:
		"""Extracted image path for row `index`."""
		return self._paths[index]

	def box_count(self, index: int) -> int:
		"""Boxes on image `index` (host-side, no readback)."""
		return self._lengths[index]

	def boxes_at(self, index: int) -> "oa.Matrix":
		"""[K, 5] f32 box slice for image `index`."""
		if self._offsets[index] == self._offsets[index + 1]:
			return oa.Matrix.new_from_f32([0, 5], [], self._engine)
		return self._boxes.slice(
			0, self._offsets[index], self._offsets[index + 1]
		)

	def image(self, index: int) -> "oa.Image":
		"""Lazily decode image `index` to a [1, 3, H, W] f32 NCHW image."""
		return oa.image.decode_file(
			str(self._paths[index]), "rgb", self._engine
		)

	def get_item(self, index: int) -> "oa.Matrix":
		"""Lazily decode image `index` as a [3, H, W] matrix."""
		item = self.image(index).as_matrix()
		return item.reshape(item.shape[1:])

	def get_sample(self, index: int):
		"""(image, [K, 5] boxes) pair for one row."""
		return self.get_item(index), self.boxes_at(index)
