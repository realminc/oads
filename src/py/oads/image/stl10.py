"""STL-10 image recognition; see `oads.dataset.MatrixDataset`."""

from .. import _native
from ..dataset import MatrixDataset


class Stl10(MatrixDataset):
	PACK = _native.pack_manifest("stl10")

	@property
	def images(self):
		"""Full image matrix on the owning engine; alias of `x`."""
		return self.x
