"""GoEmotions 28-way multi-label comment corpus.

`{train,dev,test}.tsv` materialize through `oads.pack` under
`go_emotions/1.0.0` and decode to `[N, text_len]` f32 byte windows plus
`[N, 28]` multi-hot emotion indicators (27 emotions + neutral).
`get_item` returns one `[1, text_len]` window; `get_sample` pairs it
with the `[1, 28]` indicator row.
"""

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("go_emotions")

EMOTION_COUNT = 28


class GoEmotions:
	"""[N, text_len] text windows + [N, 28] multi-hot labels."""

	def __init__(self, text, labels, split, text_len):
		self._text = text
		self._labels = labels
		self._split = split
		self._text_len = text_len

	@classmethod
	def load(
		cls,
		engine: "oa.Engine | None" = None,
		split: str = "train",
		cache_dir=None,
		normalize: bool = True,
		text_len: int = 512,
		progress=True,
	) -> "GoEmotions":
		"""Fetch, verify, decode, and upload the selected split.

		normalize=True scales bytes into [0, 1]; progress=True renders
		the download bar.
		"""
		fields = _native.load(
			PACK["name"],
			split=split,
			normalize=normalize,
			cache_dir=str(cache_dir) if cache_dir else None,
			progress=resolve(progress),
		)
		return cls(
			matrix(fields["text"], engine),
			matrix(fields["labels"], engine),
			split,
			text_len,
		)

	def __len__(self) -> int:
		return self._text.shape[0]

	@property
	def text(self) -> "oa.Matrix":
		"""[N, text_len] f32 byte windows."""
		return self._text

	@property
	def labels(self) -> "oa.Matrix":
		"""[N, 28] f32 multi-hot emotion indicators."""
		return self._labels

	@property
	def split(self) -> str:
		return self._split

	@property
	def text_len(self) -> int:
		return self._text_len

	def get_item(self, index: int) -> "oa.Matrix":
		"""[1, text_len] text window for one comment."""
		return self._text.slice(0, index, index + 1)

	def get_sample(self, index: int):
		"""(text window, [1, 28] multi-hot) pair."""
		return self.get_item(index), self._labels.slice(0, index, index + 1)
