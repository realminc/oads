"""MNIST handwritten digits; see `oads.dataset.MatrixDataset`."""

from .. import _native
from ..dataset import MatrixDataset


class Mnist(MatrixDataset):
	PACK = _native.pack_manifest("mnist")

	@property
	def images(self):
		"""Full image matrix on the owning engine; alias of `x`."""
		return self.x
