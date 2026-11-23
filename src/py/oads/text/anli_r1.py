"""Adversarial NLI round 1 context-hypothesis classification task.

The pack is materialized through `oads.pack` under `anli_r1/1.0.0` and
the selected split member decodes to one `[N, K_i]` f32 byte matrix per
text field plus `labels`. `get_item` returns the
`[1, sum(K_i)]` end-to-end field row; `get_sample` pairs it with the
label when the member carries one.
"""

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("anli_r1")
FIELDS = ["context", "hypothesis"]
CLASSES = ['entailment', 'neutral', 'contradiction']


class AnliR1:
	"""AnliR1 2-field text matricesplus `labels`."""

	def __init__(self, fields, labels, classes):
		self._fields = fields
		self._labels = labels
		self._classes = classes

	@classmethod
	def load(
		cls,
		engine: "oa.Engine | None" = None,
		split: str = "train",
		cache_dir=None,
		progress=True,
	) -> "AnliR1":
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
			[matrix(fields[name], engine) for name in FIELDS],
			matrix(fields["labels"], engine) if "labels" in fields else None,
			CLASSES,
		)

	def __len__(self) -> int:
		return self._fields[0].shape[0]

	@property
	def context(self) -> "oa.Matrix":
		"""[N, K] f32 `context` field, zero-padded."""
		return self._fields[0]

	@property
	def hypothesis(self) -> "oa.Matrix":
		"""[N, K] f32 `hypothesis` field, zero-padded."""
		return self._fields[1]

	@property
	def labels(self) -> "oa.Matrix | None":
		"""[N] f32 label matrix; `None` for unlabeled members."""
		return self._labels

	@property
	def classes(self) -> list:
		"""Class display names; label index `i` names `classes[i]`."""
		return self._classes

	def get_item(self, index: int) -> "oa.Matrix":
		"""[1, sum(K_i)] end-to-end field row for one sample."""
		import oa.matrix

		rows = [field.slice(0, index, index + 1) for field in self._fields]
		if len(rows) == 1:
			return rows[0]
		return oa.matrix.concat(rows, 1)

	def get_sample(self, index: int):
		"""(text row, label) pair; label is `None` on unlabeled members."""
		label = (
			self._labels.slice(0, index, index + 1).reshape([1])
			if self._labels is not None
			else None
		)
		return self.get_item(index), label
