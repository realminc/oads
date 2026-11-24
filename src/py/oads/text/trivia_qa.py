"""TriviaQA reading-comprehension question/answer pairs (wikipedia domain).

The pack is materialized through `oads.pack` under `trivia_qa/1.0.0`; the
selected `qa/wikipedia-*.json` member decodes `question` plus the canonical
`answer` and newline-joined `aliases` `[N, K_i]` f32 byte matrices (any alias
is a valid eval target). The `test` member carries no answers and emits
`question` only. `get_item` returns the `[1, sum(K_i)]` end-to-end field row
over whichever fields the split carries.
"""

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("trivia_qa")
FIELDS = ["question", "answer", "aliases"]


class TriviaQa:
	"""TriviaQa question/answer/aliases text matrices (test: question only)."""

	def __init__(self, fields, names):
		self._fields = fields
		self._names = names

	@classmethod
	def load(
		cls,
		engine: "oa.Engine | None" = None,
		split: str = "train",
		cache_dir=None,
		progress=True,
	) -> "TriviaQa":
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
		names = [name for name in FIELDS if name in fields]
		return cls([matrix(fields[name], engine) for name in names], names)

	def __len__(self) -> int:
		return self._fields[0].shape[0]

	@property
	def question(self) -> "oa.Matrix":
		"""[N, K] f32 `question` field, zero-padded."""
		return self._fields[0]

	@property
	def answer(self) -> "oa.Matrix | None":
		"""[N, K] f32 canonical `answer` field; `None` on `test`."""
		return self._named("answer")

	@property
	def aliases(self) -> "oa.Matrix | None":
		"""[N, K] f32 newline-joined `aliases` field; `None` on `test`."""
		return self._named("aliases")

	def _named(self, name):
		try:
			return self._fields[self._names.index(name)]
		except ValueError:
			return None

	@property
	def labels(self) -> "oa.Matrix | None":
		"""Always `None`; answers are text, not class indices."""
		return None

	def get_item(self, index: int) -> "oa.Matrix":
		"""[1, sum(K_i)] end-to-end field row for one sample."""
		import oa.matrix

		rows = [field.slice(0, index, index + 1) for field in self._fields]
		if len(rows) == 1:
			return rows[0]
		return oa.matrix.concat(rows, 1)

	def get_sample(self, index: int):
		"""(text row, None) pair; answers are text fields, not labels."""
		return self.get_item(index), None
