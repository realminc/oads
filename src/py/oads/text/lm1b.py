"""One Billion Word Benchmark tokenized sentence stream.

The pack is materialized through `oads.pack` under `lm1b/1.0.0`; the
selected shard set flattens to a `sentence` byte blob with per-row
`sentence_lengths` (offsets are the cumulative sum of lengths).
`get_item` returns the `[1, len_i]` sentence row for the requested
index.
"""

from array import array
from itertools import accumulate

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("lm1b")


class Lm1b:
	"""Lm1b flat sentence blob with per-sentence byte lengths."""

	def __init__(self, sentence, sentence_lengths, offsets):
		self._sentence = sentence
		self._sentence_lengths = sentence_lengths
		self._offsets = offsets

	@classmethod
	def load(
		cls,
		engine: "oa.Runtime | None" = None,
		split: str = "train",
		count=None,
		cache_dir=None,
		progress=True,
	) -> "Lm1b":
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
		lengths = array("f", fields["sentence_lengths"]["data"])
		offsets = [0, *accumulate(int(v) for v in lengths)]
		return cls(
			matrix(fields["sentence"], engine),
			matrix(fields["sentence_lengths"], engine),
			offsets,
		)

	def __len__(self) -> int:
		return len(self._offsets) - 1

	@property
	def sentence(self) -> "oa.Matrix":
		"""[B] f32 flat sentence byte blob on the owning engine."""
		return self._sentence

	@property
	def sentence_lengths(self) -> "oa.Matrix":
		"""[N] f32 per-sentence byte lengths; cumsum gives offsets."""
		return self._sentence_lengths

	def get_item(self, index: int) -> "oa.Matrix":
		"""[1, len_i] sentence row for one sample index."""
		start, end = self._offsets[index], self._offsets[index + 1]
		return self._sentence.slice(0, start, end)

	def get_sample(self, index: int):
		"""(sentence row, None) pair; the corpus is unlabeled."""
		return self.get_item(index), None
