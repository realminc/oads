"""Stanford Sentiment Treebank binary sentiment task (GLUE `SST-2`).

`SST-2.zip` is materialized through `oads.pack` under `glue_sst2/1.0.0` and
the selected split member decodes to padded f32 byte matrices:
`text1` plus `labels`. `get_item` returns the
`[F, K]` stacked text rows for one sample; `get_sample` pairs it
with the label or score.
"""

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("glue_sst2")
CLASSES = ['negative', 'positive']


class GlueSst2:
	"""SST-2 1-field text matrices + [N] labels."""

	def __init__(self, text1, text2, labels, classes):
		self._text1 = text1
		self._text2 = text2
		self._labels = labels
		self._classes = classes

	@classmethod
	def load(
		cls,
		engine: "oa.Engine | None" = None,
		split: str = "train",
		cache_dir=None,
		progress=True,
	) -> "GlueSst2":
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
			matrix(fields["text1"], engine),
			matrix(fields["text2"], engine) if "text2" in fields else None,
			matrix(fields["labels"], engine),
			CLASSES,
		)

	def __len__(self) -> int:
		return self._text1.shape[0]

	@property
	def text1(self) -> "oa.Matrix":
		"""[N, K] f32 first text field, zero-padded."""
		return self._text1

	@property
	def labels(self) -> "oa.Matrix":
		"""[N] f32 label or score matrix."""
		return self._labels

	@property
	def classes(self) -> list:
		"""Class display names; label index `i` names `classes[i]`."""
		return self._classes

	def get_item(self, index: int) -> "oa.Matrix":
		"""[F, K] stacked text rows for one sample."""
		import oa.matrix

		first = self._text1.slice(0, index, index + 1)
		if self._text2 is None:
			return first
		return oa.matrix.concat([first, self._text2.slice(0, index, index + 1)], 0)

	def get_sample(self, index: int):
		"""(text rows, label) pair for one sample."""
		return self.get_item(index), (
			self._labels.slice(0, index, index + 1).reshape([1])
		)
