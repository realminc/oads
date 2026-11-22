"""CIFAR-10 color objects; see `oads.dataset.MatrixDataset`."""

from .. import _native
from ..dataset import MatrixDataset


class Cifar10(MatrixDataset):
	PACK = _native.pack_manifest("cifar10")

	@property
	def images(self):
		"""Full image matrix on the owning engine; alias of `x`."""
		return self.x
