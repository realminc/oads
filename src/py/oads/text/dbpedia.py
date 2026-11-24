"""DBpedia 14-class ontology corpus; see `oads.dataset.MatrixDataset`."""

from .. import _native
from ..dataset import MatrixDataset


class Dbpedia(MatrixDataset):
	PACK = _native.pack_manifest("dbpedia")
	FEATURE = "text"

	@property
	def text(self):
		"""Full byte-window matrix on the owning engine; alias of `x`."""
		return self.x
