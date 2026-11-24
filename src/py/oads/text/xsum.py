"""Extreme Summarization (XSum) document/summary pairs.

The pack is materialized through `oads.pack` under `xsum/1.0.0`; the
EdinburghNLP manifest selects which `bbc-summary-data` members form each
split. Documents flatten to a `document` byte blob with per-row
`document_lengths` (offsets are the cumulative sum of lengths); summaries
pad to a `[N, K]` `summary` byte matrix. `get_item` returns the
`[1, len_i]` document row for the requested index.
"""

from array import array
from itertools import accumulate

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("xsum")


class Xsum:
	"""Xsum flat document blob plus padded summary matrix."""

	def __init__(self, document, document_lengths, summary, offsets):
		self._document = document
		self._document_lengths = document_lengths
		self._summary = summary
		self._offsets = offsets

	@classmethod
	def load(
		cls,
		engine: "oa.Engine | None" = None,
		split: str = "train",
		count=None,
		cache_dir=None,
		progress=True,
	) -> "Xsum":
		"""Fetch, verify, and decode the selected split.

		progress=True renders the download bar; a callable receives
		(file, fetched, total) samples instead.
		"""
		fields = _native.load(
			PACK["name"],
			split=split,
			count=count,
			cache_dir=str(cache_dir) if cache_dir else None,
			progress=resolve(progress),
		)
		lengths = array("f", fields["document_lengths"]["data"])
		offsets = [0, *accumulate(int(v) for v in lengths)]
		return cls(
			matrix(fields["document"], engine),
			matrix(fields["document_lengths"], engine),
			matrix(fields["summary"], engine),
			offsets,
		)

	def __len__(self) -> int:
		return self._summary.shape[0]

	@property
	def document(self) -> "oa.Matrix":
		"""[B] f32 flat document byte blob on the owning engine."""
		return self._document

	@property
	def document_lengths(self) -> "oa.Matrix":
		"""[N] f32 per-document byte lengths; cumsum gives blob offsets."""
		return self._document_lengths

	@property
	def summary(self) -> "oa.Matrix":
		"""[N, K] f32 padded summary byte matrix."""
		return self._summary

	def get_item(self, index: int) -> "oa.Matrix":
		"""[1, len_i] document row for one sample index."""
		start, end = self._offsets[index], self._offsets[index + 1]
		return self._document.slice(0, start, end)

	def get_sample(self, index: int):
		"""(document row, summary row) pair for one sample index."""
		return self.get_item(index), self._summary.slice(0, index, index + 1)
