"""MovieLens 1M ratings; see `oads.dataset.MatrixDataset`."""

from .. import _native
from ..dataset import MatrixDataset


class Movielens1m(MatrixDataset):
	PACK = _native.pack_manifest("movielens_1m")
	FEATURE = "features"
	LABEL = "ratings"

	@property
	def features(self):
		"""Full [N, 3] f32 (user_id, item_id, timestamp) matrix; alias of `x`."""
		return self.x

	@property
	def ratings(self):
		"""Full [N] f32 rating value matrix; alias of `labels`."""
		return self.labels
