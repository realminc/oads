"""EMNIST handwritten characters (balanced variant); see `oads.dataset.MatrixDataset`."""

from .. import _native
from ..dataset import MatrixDataset


class Emnist(MatrixDataset):
	PACK = _native.pack_manifest("emnist")

	@property
	def images(self):
		"""Full image matrix on the owning engine; alias of `x`."""
		return self.x
