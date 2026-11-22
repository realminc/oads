"""CIFAR-100 color objects with fine labels; see `oads.dataset.MatrixDataset`."""

from .. import _native
from ..dataset import MatrixDataset


class Cifar100(MatrixDataset):
	PACK = _native.pack_manifest("cifar100")

	@property
	def images(self):
		"""Full image matrix on the owning engine; alias of `x`."""
		return self.x
