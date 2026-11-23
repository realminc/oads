"""Cross-lingual NLI dev/test (XNLI).

`XNLI-1.0.zip` is materialized through `oads.pack` under `xnli/1.0.0`
and the selected split member (`xnli.dev.tsv` / `xnli.test.tsv`)
decodes to `premise`/`hypothesis` `[N, K_i]` f32 byte matrices, a `[N]`
label matrix, and a `[N]` language-id matrix indexing `LANGUAGES`.
`get_item` returns the `[1, K1 + K2]` premise+hypothesis row.
"""

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("xnli")
LANGUAGES = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
CLASSES = ['entailment', 'neutral', 'contradiction']


class Xnli:
	"""XNLI premise/hypothesis matrices + labels + language ids."""

	def __init__(self, premise, hypothesis, labels, languages):
		self._premise = premise
		self._hypothesis = hypothesis
		self._labels = labels
		self._languages = languages

	@classmethod
	def load(
		cls,
		engine: "oa.Engine | None" = None,
		split: str = "validation",
		cache_dir=None,
		progress=True,
	) -> "Xnli":
		"""Fetch, verify, and decode the selected split.

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
			matrix(fields["premise"], engine),
			matrix(fields["hypothesis"], engine),
			matrix(fields["labels"], engine),
			matrix(fields["languages"], engine),
		)

	def __len__(self) -> int:
		return self._premise.shape[0]

	@property
	def premise(self) -> "oa.Matrix":
		"""[N, K] f32 premise field, zero-padded."""
		return self._premise

	@property
	def hypothesis(self) -> "oa.Matrix":
		"""[N, K] f32 hypothesis field, zero-padded."""
		return self._hypothesis

	@property
	def labels(self) -> "oa.Matrix":
		"""[N] f32 label matrix."""
		return self._labels

	@property
	def languages(self) -> "oa.Matrix":
		"""[N] f32 language ids; index `i` names `LANGUAGES[i]`."""
		return self._languages

	@property
	def classes(self) -> list:
		"""Class display names; label index `i` names `classes[i]`."""
		return CLASSES

	def get_item(self, index: int) -> "oa.Matrix":
		"""[1, K1 + K2] premise+hypothesis byte row."""
		import oa.matrix

		return oa.matrix.concat(
			[
				self._premise.slice(0, index, index + 1),
				self._hypothesis.slice(0, index, index + 1),
			],
			1,
		)

	def get_sample(self, index: int):
		"""(premise+hypothesis row, label) pair."""
		return self.get_item(index), (
			self._labels.slice(0, index, index + 1).reshape([1])
		)
