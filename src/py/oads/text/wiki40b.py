"""Wiki-40B cleaned English Wikipedia article corpus.

The pack is materialized through `oads.pack` under `wiki40b/1.0.0`;
140 TFRecord shards carry `text` and `wikidata_id` features. Articles
flatten to a `text` byte blob with per-row `article_lengths` and a
parallel `wikidata_id` blob (offsets are the cumulative sum of
lengths). `get_item` returns the `[1, len_i]` article row for the
requested index.
"""

from array import array
from itertools import accumulate

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("wiki40b")


class Wiki40B:
	"""Wiki-40B flat article blob with per-article byte lengths."""

	def __init__(self, text, article_lengths, wikidata_id, wikidata_id_lengths, offsets):
		self._text = text
		self._article_lengths = article_lengths
		self._wikidata_id = wikidata_id
		self._wikidata_id_lengths = wikidata_id_lengths
		self._offsets = offsets

	@classmethod
	def load(
		cls,
		engine: "oa.Runtime | None" = None,
		split: str = "train",
		count=None,
		cache_dir=None,
		progress=True,
	) -> "Wiki40B":
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
		lengths = array("f", fields["article_lengths"]["data"])
		offsets = [0, *accumulate(int(v) for v in lengths)]
		return cls(
			matrix(fields["text"], engine),
			matrix(fields["article_lengths"], engine),
			matrix(fields["wikidata_id"], engine),
			matrix(fields["wikidata_id_lengths"], engine),
			offsets,
		)

	def __len__(self) -> int:
		return len(self._offsets) - 1

	@property
	def text(self) -> "oa.Matrix":
		"""[B] f32 flat article byte blob on the owning engine."""
		return self._text

	@property
	def article_lengths(self) -> "oa.Matrix":
		"""[N] f32 per-article byte lengths; cumsum gives offsets."""
		return self._article_lengths

	@property
	def wikidata_id(self) -> "oa.Matrix":
		"""[B] f32 flat Wikidata id byte blob on the owning engine."""
		return self._wikidata_id

	@property
	def wikidata_id_lengths(self) -> "oa.Matrix":
		"""[N] f32 per-article Wikidata id byte lengths."""
		return self._wikidata_id_lengths

	def get_item(self, index: int) -> "oa.Matrix":
		"""[1, len_i] article row for one sample index."""
		start, end = self._offsets[index], self._offsets[index + 1]
		return self._text.slice(0, start, end)

	def get_sample(self, index: int):
		"""(article row, None) pair; the corpus is unlabeled."""
		return self.get_item(index), None
