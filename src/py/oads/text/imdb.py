"""IMDb sentiment reviews; see `oads.dataset.MatrixDataset`."""

from .. import _native
from ..dataset import MatrixDataset


class Imdb(MatrixDataset):
	PACK = _native.pack_manifest("imdb")
	FEATURE = "text"

	@property
	def text(self):
		"""Full byte-window matrix on the owning engine; alias of `x`."""
		return self.x
