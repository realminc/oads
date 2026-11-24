"""UCI Covertype forest cover type; see `oads.dataset.MatrixDataset`."""

from pathlib import Path

from .. import _native
from .._progress import resolve
from ..dataset import MatrixDataset, matrix


class Covertype(MatrixDataset):
	PACK = _native.pack_manifest("covertype")
	FEATURE = "features"

	@classmethod
	def load(cls, engine=None, cache_dir=None, normalize: bool = True, progress=True):
		"""Fetch, verify, decode, and upload the full pack.

	normalize=True scales features into the pack's canonical unit
	range; normalize=False keeps raw f32 values. progress=True renders
	the download bar.
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
		"""Full feature matrix on the owning engine; alias of `x`."""
		return self.x
