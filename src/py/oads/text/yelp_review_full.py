"""Yelp full 5-star review corpus; see `oads.dataset.MatrixDataset`."""

from .. import _native
from ..dataset import MatrixDataset


class YelpReviewFull(MatrixDataset):
	PACK = _native.pack_manifest("yelp_review_full")
	FEATURE = "text"

	@property
	def text(self):
		"""Full byte-window matrix on the owning engine; alias of `x`."""
		return self.x
