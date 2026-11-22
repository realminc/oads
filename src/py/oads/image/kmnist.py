"""KMNIST Kuzushiji characters; see `oads.dataset.MatrixDataset`."""

from .. import _native
from ..dataset import MatrixDataset


class KMnist(MatrixDataset):
	PACK = _native.pack_manifest("kmnist")

	@property
	def images(self):
		"""Full image matrix on the owning engine; alias of `x`."""
		return self.x
