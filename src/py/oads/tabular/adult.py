"""UCI Adult census income; see `oads.dataset.MatrixDataset`."""

from .. import _native
from ..dataset import MatrixDataset


class Adult(MatrixDataset):
	PACK = _native.pack_manifest("adult")
	FEATURE = "features"

	@property
	def features(self):
		"""Full feature matrix on the owning engine; alias of `x`."""
		return self.x
