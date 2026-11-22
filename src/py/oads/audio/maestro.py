"""MAESTRO v3 MIDI performances as a windowed device-resident dataset.

maestro-v3.0.0-midi.zip is materialized through `oads.pack` under
maestro/3.0.0; each performance decodes through the native SMF parser into
(pitch, step, duration) events in seconds under the file's tempo map and
concatenates into one [E, 3] f32 matrix. The pack CSV owns the official
train/validation/test assignment — 'train' and 'test' are exposed here and
'validation' through the typed Rust config.

A dataset item is a fixed [seq_len, 3] window sliced on-device from the
event matrix; windows never cross a performance boundary, so items are
contiguous views with no event duplication. get_sample pairs the window
with the same-length window shifted one event — the next-event prediction
contract of the music-generation tutorial.
"""

from array import array

import oa

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("maestro")

SEQ_LEN = 25


class Maestro:
	"""[seq_len, 3] event windows over one resident [E, 3] event matrix."""

	def __init__(
		self,
		notes: "oa.Matrix",
		offsets: "array",
		seq_len: int,
		split: str,
	):
		self._notes = notes
		self._offsets = offsets
		self._seq_len = seq_len
		self._split = split

	@classmethod
	def load(
		cls,
		engine: "oa.Engine | None" = None,
		split: str = "train",
		cache_dir=None,
		progress=True,
	) -> "Maestro":
		"""Fetch, verify, decode, and upload the selected official split.

		progress=True renders the download bar; a callable receives
		(file, fetched, total) samples instead.
		"""
		fields = _native.load(
			PACK["name"],
			split=split,
			cache_dir=str(cache_dir) if cache_dir else None,
			progress=resolve(progress),
		)
		offsets = array("f", fields["windows"]["data"])
		return cls(
			matrix(fields["notes"], engine),
			offsets,
			fields["seq_len"],
			split,
		)

	def __len__(self) -> int:
		return len(self._offsets)

	@property
	def notes(self) -> "oa.Matrix":
		"""Resident [E, 3] (pitch, step, duration) event matrix."""
		return self._notes

	@property
	def seq_len(self) -> int:
		return self._seq_len

	@property
	def split(self) -> str:
		return self._split

	def get_item(self, index: int) -> "oa.Matrix":
		"""[seq_len, 3] event window for one sample index."""
		self._check(index)
		offset = int(self._offsets[index])
		return self._notes.slice(0, offset, offset + self._seq_len)

	def get_sample(self, index: int):
		"""(x, y): window and the one-event-shifted target window."""
		self._check(index)
		offset = int(self._offsets[index])
		return (
			self._notes.slice(0, offset, offset + self._seq_len),
			self._notes.slice(0, offset + 1, offset + 1 + self._seq_len),
		)

	def _check(self, index: int) -> None:
		if index < 0 or index >= len(self):
			raise IndexError("maestro window index is outside the dataset")
