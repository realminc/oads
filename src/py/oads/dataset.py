"""Shared labeled-matrix dataset base over `oa.data.MatrixDataset`.

Subclasses carry a class-level PACK manifest dict; load() materializes the
pack and decodes it on the host through `oads._native`, then uploads a
rank-N f32 feature matrix and a [N] f32 label matrix into the native
`oa.data` contract so per-sample slices collate the same way in both
languages.
"""

from array import array
from pathlib import Path

import oa
import oa.data

from . import _native
from ._progress import resolve


def matrix(field: dict, engine) -> "oa.Matrix":
	"""Upload a native matrix field {"shape": [...], "data": f32 bytes}."""
	values = array("f", field["data"])
	return oa.Matrix.new_from_f32(field["shape"], values, engine)


class MatrixDataset:
	"""Device-resident labeled dataset over the native `oa.data` contract.

	`x` is a rank-N f32 feature matrix and `labels` is [N] f32 class indices
	on the load engine; `len`, `get_item`, and `get_sample` delegate to
	`oa.data.MatrixDataset`. Subclasses expose the domain-named alias
	(`images`, `samples`) for x.
	"""

	PACK: dict = {}
	FEATURE = "images"
	LABEL = "labels"

	@classmethod
	def load(
		cls,
		engine: "oa.Engine | None" = None,
		split: str = "train",
		cache_dir=None,
		normalize: bool = True,
		progress=True,
	) -> "MatrixDataset":
		"""Fetch, verify, decode, and upload the selected split.

		engine=None resolves the calling thread's default engine
		(oa.Engine.current()); pass an oa.Engine to pin one explicitly.
		cache_dir=None resolves OA_DATA_DIR then the platform cache.
		normalize=True scales features into the pack's canonical unit
		range; normalize=False keeps raw f32 values. progress=True renders
		the download bar; a callable receives (file, fetched, total)
		samples instead.
		"""
		fields = _native.load(
			cls.PACK["name"],
			split=split,
			normalize=normalize,
			cache_dir=str(Path(cache_dir)) if cache_dir else None,
			progress=resolve(progress),
		)
		return cls(
			matrix(fields[cls.FEATURE], engine),
			matrix(fields[cls.LABEL], engine),
			split,
		)

	def __init__(self, x: "oa.Matrix", labels: "oa.Matrix", split: str):
		self._ds = oa.data.MatrixDataset(x, labels)
		self._split = split

	def __len__(self) -> int:
		return len(self._ds)

	@property
	def x(self) -> "oa.Matrix":
		"""Full [N, ...] feature matrix on the owning engine."""
		return self._ds.x

	@property
	def labels(self) -> "oa.Matrix":
		"""Full [N] f32 class-index matrix."""
		return self._ds.y

	@property
	def split(self) -> str:
		return self._split

	def get_item(self, index: int) -> "oa.Matrix":
		"""Feature slice for one sample index."""
		return self._ds.get_item(index)

	def get_sample(self, index: int):
		"""(x, y) pair: feature slice and [1] label slice."""
		return self._ds.get_sample(index)
