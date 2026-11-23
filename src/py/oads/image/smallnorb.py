"""small NORB stereo-pair object recognition.

`{training,testing}-{dat,cat,info}.gz` members materialize under
`smallnorb/1.0.0` and decode to `[N, 2, 96, 96]` f32 stereo images,
`[N]` category labels (5 classes), and `[N, 4]` instance metadata
(instance, elevation, azimuth, lighting).
"""

from .. import _native
from .._progress import resolve
from ..dataset import MatrixDataset, matrix

CLASSES = ["animal", "human", "airplane", "truck", "car"]


class Smallnorb(MatrixDataset):
	PACK = _native.pack_manifest("smallnorb")

	def __init__(self, x, labels, split, info):
		super().__init__(x, labels, split)
		self._info = info

	@classmethod
	def load(
		cls,
		engine: "oa.Engine | None" = None,
		split: str = "train",
		cache_dir=None,
		normalize: bool = True,
		progress=True,
	) -> "Smallnorb":
		"""Fetch, verify, decode, and upload the selected split.

		Same resolution rules as `MatrixDataset.load`; also uploads the
		[N, 4] instance-metadata matrix as `info`.
		"""
		fields = _native.load(
			cls.PACK["name"],
			split=split,
			normalize=normalize,
			cache_dir=str(cache_dir) if cache_dir else None,
			progress=resolve(progress),
		)
		return cls(
			matrix(fields["images"], engine),
			matrix(fields["labels"], engine),
			split,
			matrix(fields["info"], engine),
		)

	@property
	def images(self):
		"""Full `[N, 2, 96, 96]` f32 stereo image matrix; alias of `x`."""
		return self.x

	@property
	def info(self) -> "oa.Matrix":
		"""`[N, 4]` f32 instance/elevation/azimuth/lighting matrix."""
		return self._info

	@property
	def classes(self) -> list:
		"""Canonical 5-class names; `labels` values index them."""
		return CLASSES
