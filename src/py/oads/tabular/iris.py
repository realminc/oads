"""Fisher's Iris flower data; see `oads.dataset.MatrixDataset`.

features is [150, 4] f32 (sepal/petal length and width) and labels is
[150] f32 class indices into CLASSES. The pack carries all 150 rows
unsplit; partition with oa.data.random_split.
"""

from pathlib import Path

from .. import _native
from .._progress import resolve
from ..dataset import MatrixDataset, matrix


class Iris(MatrixDataset):
	PACK = _native.pack_manifest("iris")
	FEATURE = "features"
	CLASSES = ("setosa", "versicolor", "virginica")

	@classmethod
	def load(cls, engine=None, cache_dir=None, normalize: bool = True, progress=True):
		"""Fetch, verify, decode, and upload all 150 rows.

normalize=True scales each column to [0, 1] by its min-max range;
normalize=False keeps centimetres. progress=True renders the download
bar.
		"""
		fields = _native.load(
			cls.PACK["name"],
			# The pack carries one unsplit set; the native split argument
			# only accepts train/test and is ignored for this loader.
			split="train",
			normalize=normalize,
			cache_dir=str(Path(cache_dir)) if cache_dir else None,
			progress=resolve(progress),
		)
		return cls(
			matrix(fields[cls.FEATURE], engine),
			matrix(fields[cls.LABEL], engine),
			"all",
		)

	@property
	def features(self):
		"""Full [150, 4] f32 feature matrix; alias of `x`."""
		return self.x
