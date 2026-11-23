"""SQuAD 2.0 reading-comprehension corpus.

`{train,dev}-v2.0.json` materialize through `oads.pack` under
`squad_v2/1.0.0`. Text columns are variable-length, so each decodes to
a flat `[B]` f32 byte blob plus `offsets`/`lengths` tables; contexts
additionally deduplicate to `P` unique paragraphs indexed by `[N]`
`context_index`. `answer_start` and `is_impossible` are `[N]` (-1 marks
unanswerable questions). `get_item` concatenates one question and
context slice to `[Kq_i + Kc_i]`; `get_sample` pairs it with
`answer_start`.
"""

from array import array

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("squad_v2")


def _field_matrix(fields, name, engine):
	return matrix(fields[name], engine)


def _span(fields, name):
	starts = array("f", fields[f"{name}_offsets"]["data"])
	lens = array("f", fields[f"{name}_lengths"]["data"])
	return list(zip(starts, lens))


class SquadV2:
	"""SQuAD 2.0 flat text blobs + answer spans."""

	def __init__(self, fields, engine):
		self._question_bytes = _field_matrix(fields, "question_bytes", engine)
		self._question_offsets = _field_matrix(
			fields, "question_offsets", engine
		)
		self._question_lengths = _field_matrix(
			fields, "question_lengths", engine
		)
		self._context_bytes = _field_matrix(fields, "context_bytes", engine)
		self._context_offsets = _field_matrix(
			fields, "context_offsets", engine
		)
		self._context_lengths = _field_matrix(
			fields, "context_lengths", engine
		)
		self._context_index = _field_matrix(fields, "context_index", engine)
		self._answer_bytes = _field_matrix(fields, "answer_bytes", engine)
		self._answer_offsets = _field_matrix(fields, "answer_offsets", engine)
		self._answer_lengths = _field_matrix(fields, "answer_lengths", engine)
		self._answer_start = _field_matrix(fields, "answer_start", engine)
		self._impossible = _field_matrix(fields, "is_impossible", engine)
		self._question_span = _span(fields, "question")
		self._context_span = _span(fields, "context")
		self._context_lookup = [
			int(v) for v in array("f", fields["context_index"]["data"])
		]

	@classmethod
	def load(
		cls,
		engine: "oa.Engine | None" = None,
		split: str = "train",
		cache_dir=None,
		progress=True,
	) -> "SquadV2":
		"""Fetch, verify, decode, and upload the selected split.

		progress=True renders the download bar; a callable receives
		(file, fetched, total) samples instead.
		"""
		fields = _native.load(
			PACK["name"],
			split=split,
			cache_dir=str(cache_dir) if cache_dir else None,
			progress=resolve(progress),
		)
		return cls(fields, engine)

	def __len__(self) -> int:
		return self._question_offsets.shape[0]

	@property
	def question_bytes(self) -> "oa.Matrix":
		"""`[Bq]` f32 flat question byte blob."""
		return self._question_bytes

	@property
	def question_offsets(self) -> "oa.Matrix":
		"""`[N]` f32 byte offsets of each question."""
		return self._question_offsets

	@property
	def question_lengths(self) -> "oa.Matrix":
		"""`[N]` f32 byte lengths of each question."""
		return self._question_lengths

	@property
	def context_bytes(self) -> "oa.Matrix":
		"""`[Bc]` f32 flat deduplicated context byte blob."""
		return self._context_bytes

	@property
	def context_offsets(self) -> "oa.Matrix":
		"""`[P]` f32 byte offsets of each paragraph."""
		return self._context_offsets

	@property
	def context_lengths(self) -> "oa.Matrix":
		"""`[P]` f32 byte lengths of each paragraph."""
		return self._context_lengths

	@property
	def context_index(self) -> "oa.Matrix":
		"""`[N]` f32 paragraph index per question."""
		return self._context_index

	@property
	def answer_bytes(self) -> "oa.Matrix":
		"""`[Ba]` f32 flat answer byte blob."""
		return self._answer_bytes

	@property
	def answer_offsets(self) -> "oa.Matrix":
		"""`[N]` f32 byte offsets of each answer."""
		return self._answer_offsets

	@property
	def answer_lengths(self) -> "oa.Matrix":
		"""`[N]` f32 byte lengths of each answer."""
		return self._answer_lengths

	@property
	def answer_start(self) -> "oa.Matrix":
		"""`[N]` f32 context byte offsets; -1 for impossible."""
		return self._answer_start

	@property
	def is_impossible(self) -> "oa.Matrix":
		"""`[N]` f32 unanswerable indicators."""
		return self._impossible

	def get_item(self, index: int) -> "oa.Matrix":
		"""`[Kq_i + Kc_i]` concatenated question+context bytes."""
		import oa.matrix

		qs, ql = self._question_span[index]
		cs, cl = self._context_span[self._context_lookup[index]]
		q = self._question_bytes.slice(0, int(qs), int(qs + ql))
		c = self._context_bytes.slice(0, int(cs), int(cs + cl))
		return oa.matrix.concat([q, c], 0)

	def get_sample(self, index: int):
		"""(x, y) pair: concat bytes and `[1]` answer_start slice."""
		return self.get_item(index), self._answer_start.slice(
			0, index, index + 1
		)
