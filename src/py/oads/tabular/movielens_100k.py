"""MovieLens 100k ratings; see `oads.dataset.MatrixDataset`."""

from .. import _native
from ..dataset import MatrixDataset


class Movielens100k(MatrixDataset):
	PACK = _native.pack_manifest("movielens_100k")
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
