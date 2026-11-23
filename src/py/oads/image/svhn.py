"""Street View House Numbers (SVHN); see `oads.dataset.MatrixDataset`.

`{train,test,extra}_32x32.mat` materialize under `svhn/1.0.0` and decode
to `[N, 3, 32, 32]` f32 images plus `[N]` digit labels (0-9; the
archive's label 10 maps to 0). The `extra` split is not reachable
through the split-string bridge and remains Rust-only.
"""

from .. import _native
from ..dataset import MatrixDataset


class Svhn(MatrixDataset):
	PACK = _native.pack_manifest("svhn")

	@property
	def images(self):
		"""Full image matrix on the owning engine; alias of `x`."""
		return self.x
