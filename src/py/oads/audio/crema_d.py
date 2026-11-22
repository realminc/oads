"""CREMA-D speech-emotion clips.

crema-d-mirror-main.tar.gz is materialized through `oads.pack` under
crema_d/1.0.0; the AudioWAV members decode to a [N, 1, T] f32 waveform
matrix zero-padded to the longest clip, a [N] f32 matrix of true sample
counts at 16,000 Hz, and [N] f32 emotion indices into CLASSES.

The corpus ships no official split; 'train', 'validation', and 'test'
are a deterministic speaker-disjoint partition derived from sorted actor
ids so no actor crosses splits.
"""

import oa

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("crema_d")

CLASSES = ("NEU", "HAP", "SAD", "ANG", "FEA", "DIS")
SAMPLE_RATE = 16000


class CremaD:
	"""[N, 1, T] padded waveforms + [N] lengths/labels per split."""

	def __init__(
		self,
		waveforms: "oa.Matrix",
		lengths: "oa.Matrix",
		labels: "oa.Matrix",
		split: str,
	):
		self._waveforms = waveforms
		self._lengths = lengths
		self._labels = labels
		self._split = split

	@classmethod
	def load(
		cls,
		engine: "oa.Engine | None" = None,
		split: str = "train",
		cache_dir=None,
		progress=True,
	) -> "CremaD":
		"""Fetch, verify, decode, and upload the selected split.

		split is 'train', 'test', or (Rust only) 'validation'.
		progress=True renders the download bar; a callable receives
		(file, fetched, total) samples instead.
		"""
		fields = _native.load(
			PACK["name"],
			split=split,
			cache_dir=str(cache_dir) if cache_dir else None,
			progress=resolve(progress),
		)
		return cls(
			matrix(fields["waveforms"], engine),
			matrix(fields["lengths"], engine),
			matrix(fields["labels"], engine),
			split,
		)

	def __len__(self) -> int:
		return self._labels.shape[0]

	@property
	def waveforms(self) -> "oa.Matrix":
		"""[N, 1, T] f32 zero-padded waveforms."""
		return self._waveforms

	@property
	def lengths(self) -> "oa.Matrix":
		"""[N] f32 true clip sample counts."""
		return self._lengths

	@property
	def labels(self) -> "oa.Matrix":
		"""[N] f32 emotion indices into CLASSES."""
		return self._labels

	@property
	def split(self) -> str:
		return self._split

	def get_item(self, index: int) -> "oa.Matrix":
		"""[1, T] padded waveform for one clip."""
		return self._waveforms.slice(0, index, index + 1).reshape(
			list(self._waveforms.shape[1:])
		)

	def get_sample(self, index: int):
		"""(waveform, label) pair for one clip."""
		return self.get_item(index), self._labels.slice(0, index, index + 1)
