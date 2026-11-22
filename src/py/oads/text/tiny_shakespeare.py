"""Tiny Shakespeare as a device-resident character dataset.

The ~1.1 MB input.txt is materialized through `oads.pack` under
tiny_shakespeare/1.0.0 and uploaded once as a [N] f32 matrix of byte values
0..255. A sample is a seq_len-byte token window whose label is the same
window shifted one byte ahead — the standard character language-model
contract.
"""

import oa

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("tiny_shakespeare")

DEFAULT_SEQ_LEN = 128


class TinyShakespeare:
	def __init__(self, tokens: "oa.Matrix", text: str, seq_len: int):
		self._tokens = tokens
		self._text = text
		self._seq_len = seq_len

	@classmethod
	def load(
		cls,
		engine: "oa.Engine | None" = None,
		seq_len: int = DEFAULT_SEQ_LEN,
		cache_dir=None,
		progress=True,
	) -> "TinyShakespeare":
		if seq_len < 1:
			raise ValueError("tiny_shakespeare seq_len must be at least 1")
		fields = _native.load(
			PACK["name"],
			cache_dir=str(cache_dir) if cache_dir else None,
			progress=resolve(progress),
		)
		count = fields["tokens"]["shape"][0]
		if count <= seq_len:
			raise ValueError(
				f"tiny_shakespeare seq_len {seq_len} leaves no sample in {count} bytes"
			)
		tokens = matrix(fields["tokens"], engine)
		return cls(tokens, fields["text"], seq_len)

	def __len__(self) -> int:
		return self._tokens.shape[0] - self._seq_len

	@property
	def tokens(self) -> "oa.Matrix":
		"""Full [N] f32 byte-token matrix on the owning engine."""
		return self._tokens

	@property
	def text(self) -> str:
		"""The decoded UTF-8 corpus held on the host."""
		return self._text

	@property
	def seq_len(self) -> int:
		return self._seq_len

	def get_item(self, index: int) -> "oa.Matrix":
		"""[seq_len] token window starting at index."""
		self._check(index)
		return self._tokens.slice(0, index, index + self._seq_len)

	def get_sample(self, index: int):
		"""(x, y) pair: [seq_len] window and the one-byte-shifted label window."""
		self._check(index)
		end = index + self._seq_len
		return (
			self._tokens.slice(0, index, end),
			self._tokens.slice(0, index + 1, end + 1),
		)

	def _check(self, index: int) -> None:
		if index < 0 or index >= len(self):
			raise IndexError("tiny_shakespeare index is outside the dataset")
