"""Yelp review polarity; see `oads.dataset.MatrixDataset`."""

from .. import _native
from ..dataset import MatrixDataset


class YelpPolarity(MatrixDataset):
	PACK = _native.pack_manifest("yelp_polarity")
	FEATURE = "text"

	@property
	def text(self):
		"""Full byte-window matrix on the owning engine; alias of `x`."""
		return self.x
