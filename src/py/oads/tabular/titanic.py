"""Titanic passenger survival table; see `oads.dataset.MatrixDataset`.

features is [891, 9] f32 — pclass, sex, age, sibsp, parch, fare, and
one-hot embarked — and labels is [891] f32 survival flags. Missing age
fills with the column median; missing embarked fills S. The pack carries
all 891 rows unsplit; partition with oa.data.random_split.
"""

from pathlib import Path

from .. import _native
from .._progress import resolve
from ..dataset import MatrixDataset, matrix


class Titanic(MatrixDataset):
	PACK = _native.pack_manifest("titanic")
	FEATURE = "features"

	@classmethod
	def load(cls, engine=None, cache_dir=None, normalize: bool = True, progress=True):
		"""Fetch, verify, decode, and upload all 891 rows.

normalize=True scales each column to [0, 1] by its min-max range;
normalize=False keeps encoded units. progress=True renders the
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
		"""Full [891, 9] f32 feature matrix; alias of `x`."""
		return self.x
