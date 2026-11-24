"""Montesinho forest-fire burn area; see `oads.dataset.MatrixDataset`.

features is [517, 12] f32 (cell X/Y, month and weekday ordinals,
FFMC/DMC/DC/ISI, temp/RH/wind/rain) and labels is [517] f32 burned
hectares. The pack carries all 517 rows unsplit; partition with
oa.data.random_split.
"""

from pathlib import Path

from .. import _native
from .._progress import resolve
from ..dataset import MatrixDataset, matrix


class ForestFires(MatrixDataset):
	PACK = _native.pack_manifest("forest_fires")
	FEATURE = "features"

	@classmethod
	def load(cls, engine=None, cache_dir=None, normalize: bool = True, progress=True):
		"""Fetch, verify, decode, and upload all 517 rows.

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
		"""Full [517, 12] f32 feature matrix; alias of `x`."""
		return self.x
