"""Waves yes/no speech corpus.

`waves_yesno.tar.gz` materializes under `waves_yesno/1.0.0` and decodes
to `[60, 1, T]` f32 zero-padded 8 kHz mono waveforms, `[60]` sample
counts, and `[60, 8]` per-token yes/no labels parsed from the filename
bits (1 = yes). Labels are multi-hot, so this is not a
`oads.dataset.MatrixDataset` — `get_item` returns one `[1, T]` row and
`get_sample` pairs it with the `[8]` label row.
"""

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("waves_yesno")
SAMPLE_RATE = 8000


class WavesYesno:
	"""Yes/no `[N, 1, T]` padded waveforms + `[N, 8]` labels."""

	def __init__(self, waveforms, lengths, labels):
		self._waveforms = waveforms
		self._lengths = lengths
		self._labels = labels

	@classmethod
	def load(
		cls,
		engine: "oa.Engine | None" = None,
		cache_dir=None,
		progress=True,
	) -> "WavesYesno":
		"""Fetch, verify, decode, and upload the pack.

		progress=True renders the download bar; a callable receives
		(file, fetched, total) samples instead.
		"""
		fields = _native.load(
			PACK["name"],
			split="train",
			cache_dir=str(cache_dir) if cache_dir else None,
			progress=resolve(progress),
		)
		return cls(
			matrix(fields["waveforms"], engine),
			matrix(fields["lengths"], engine),
			matrix(fields["labels"], engine),
		)

	def __len__(self) -> int:
		return self._waveforms.shape[0]

	@property
	def waveforms(self) -> "oa.Matrix":
		"""`[N, 1, T]` f32 zero-padded waveform matrix."""
		return self._waveforms

	@property
	def lengths(self) -> "oa.Matrix":
		"""`[N]` f32 per-clip sample counts at SAMPLE_RATE."""
		return self._lengths

	@property
	def labels(self) -> "oa.Matrix":
		"""`[N, 8]` f32 per-token yes indicators."""
		return self._labels

	def get_item(self, index: int) -> "oa.Matrix":
		"""Clip `index` as a `[1, T]` f32 waveform."""
		shape = self._waveforms.shape
		return self._waveforms.slice(0, index, index + 1).reshape(
			shape[1:]
		)

	def get_sample(self, index: int):
		"""(x, y) pair: `[1, T]` waveform and `[1, 8]` label row."""
		return self.get_item(index), self._labels.slice(
			0, index, index + 1
		)

	def waveform(self, index: int) -> "oa.Matrix":
		"""Clip `index` as a `[1, T]` f32 waveform."""
		return self.get_item(index)
