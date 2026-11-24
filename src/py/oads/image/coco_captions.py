"""MS COCO 2017 captions — five UTF-8 captions per image.

Shares the `coco` pack's `train2017`/`val2017` image zips plus
`annotations_trainval2017.zip`; each sample is one caption paired with
its source image (index order is caption order, five consecutive rows
per image). Images decode lazily to `[3, H, W]` f32 NCHW; captions
slice out of the resident `caption_bytes` f32 blob.
"""

from array import array
from itertools import accumulate
from pathlib import Path

import oa

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("coco_captions")


class CocoCaptions:
	"""Lazy [3, H, W] images + [K] UTF-8 caption byte slices."""

	def __init__(self, caption_bytes, lengths, offsets, image_ids, paths, engine):
		self._caption_bytes = caption_bytes
		self._lengths = lengths
		self._offsets = offsets
		self._image_ids = image_ids
		self._paths = paths
		self._engine = engine

	@classmethod
	def load(
		cls,
		engine: "oa.Engine | None" = None,
		split: str = "train",
		cache_dir=None,
		progress=True,
	) -> "CocoCaptions":
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
		lengths = [int(v) for v in array("f", fields["caption_lengths"]["data"])]
		offsets = [0, *accumulate(lengths)]
		image_ids = [int(v) for v in array("f", fields["image_ids"]["data"])]
		return cls(
			matrix(fields["caption_bytes"], resolved),
			lengths,
			offsets,
			image_ids,
			paths,
			resolved,
		)

	def __len__(self) -> int:
		return len(self._lengths)

	@property
	def caption_bytes(self) -> "oa.Matrix":
		"""[B] f32 flat UTF-8 caption byte blob."""
		return self._caption_bytes

	def image_id(self, index: int) -> int:
		"""Source COCO image id for caption row `index`."""
		return self._image_ids[index]

	def path(self, index: int) -> Path:
		"""Extracted image path for the image behind caption `index`."""
		return self._paths[index]

	def caption(self, index: int) -> "oa.Matrix":
		"""[K] f32 UTF-8 caption byte slice for row `index`."""
		return self._caption_bytes.slice(
			0, self._offsets[index], self._offsets[index + 1]
		)

	def caption_text(self, index: int) -> str:
		"""Decoded caption string for row `index` (host readback)."""
		data = self.caption(index).read("f32")
		return bytes(int(v) for v in data).decode("utf-8")

	def image(self, index: int) -> "oa.Image":
		"""Lazily decode the image behind caption `index` to [1, 3, H, W]."""
		return oa.image.decode_file(
			str(self._paths[index]), "rgb", self._engine
		)

	def get_item(self, index: int) -> "oa.Matrix":
		"""Lazily decode the image behind caption `index` as [3, H, W]."""
		item = self.image(index).as_matrix()
		return item.reshape(item.shape[1:])

	def get_sample(self, index: int):
		"""(image, caption bytes) pair for one caption row."""
		return self.get_item(index), self.caption(index)
