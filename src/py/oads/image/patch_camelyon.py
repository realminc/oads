"""PatchCamelyon histopathology patches; see `oads.dataset.MatrixDataset`."""

from .. import _native
from ..dataset import MatrixDataset


class PatchCamelyon(MatrixDataset):
	PACK = _native.pack_manifest("patch_camelyon")

	@property
	def images(self):
		"""Full patch matrix on the owning engine; alias of `x`."""
		return self.x
