"""Fashion-MNIST clothing items; see `oads.dataset.MatrixDataset`."""

from .. import _native
from ..dataset import MatrixDataset


class FashionMnist(MatrixDataset):
	PACK = _native.pack_manifest("fashion_mnist")

	@property
	def images(self):
		"""Full image matrix on the owning engine; alias of `x`."""
		return self.x
