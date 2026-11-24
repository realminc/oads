"""Statlog German credit risk; see `oads.dataset.MatrixDataset`.

features is [1000, 24] f32 (numeric recoding of the 20-attribute set)
and labels is [1000] f32 risk indices into CLASSES (0 good, 1 bad).
The pack carries all 1000 rows unsplit; partition with
oa.data.random_split.
"""

from pathlib import Path

from .. import _native
from .._progress import resolve
from ..dataset import MatrixDataset, matrix


class GermanCredit(MatrixDataset):
	PACK = _native.pack_manifest("german_credit")
	FEATURE = "features"
	CLASSES = ("good", "bad")

	@classmethod
	def load(cls, engine=None, cache_dir=None, normalize: bool = True, progress=True):
		"""Fetch, verify, decode, and upload all 1000 rows.

normalize=True scales each column to [0, 1] by its min-max range;
normalize=False keeps source units. progress=True renders the
download bar.
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
		"""Full [1000, 24] f32 feature matrix; alias of `x`."""
		return self.x
