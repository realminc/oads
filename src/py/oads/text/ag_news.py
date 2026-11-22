"""AG News topic classification; see `oads.dataset.MatrixDataset`."""

from .. import _native
from ..dataset import MatrixDataset


class AgNews(MatrixDataset):
	PACK = _native.pack_manifest("ag_news")
	FEATURE = "text"

	@property
	def text(self):
		"""Full byte-window matrix on the owning engine; alias of `x`."""
		return self.x
