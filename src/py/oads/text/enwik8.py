"""enwik8 as a device-resident byte-level dataset.

`enwik8.zip` materializes through `oads.pack` under `enwik8/1.0.0`;
its single `enwik8` member carries the first 100,000,000 bytes of a
Wikipedia XML dump and the selected byte-range split uploads once as an
`[N]` f32 matrix of byte values 0..255 (`train` is the first
90,000,000, `validation` the next 5,000,000, `test` the final
5,000,000). A sample is a `seq_len`-byte token window whose label is
the same window shifted one byte ahead — the standard character
language-model contract.
"""

import oa

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("enwik8")

DEFAULT_SEQ_LEN = 128


class Enwik8:
	"""[N] f32 byte tokens + `seq_len` window samples."""

	def __init__(self, tokens: "oa.Matrix", seq_len: int):
		self._tokens = tokens
		self._seq_len = seq_len

	@classmethod
	def load(
		cls,
		engine: "oa.Runtime | None" = None,
		seq_len: int = DEFAULT_SEQ_LEN,
		split: str = "train",
		cache_dir=None,
		progress=True,
	) -> "Enwik8":
		if seq_len < 1:
			raise ValueError("enwik8 seq_len must be at least 1")
		fields = _native.load(
			PACK["name"],
			split=split,
			cache_dir=str(cache_dir) if cache_dir else None,
			progress=resolve(progress),
		)
		count = fields["tokens"]["shape"][0]
		if count <= seq_len:
			raise ValueError(
				f"enwik8 seq_len {seq_len} leaves no sample in {count} bytes"
			)
		return cls(matrix(fields["tokens"], engine), seq_len)

	def __len__(self) -> int:
		return self._tokens.shape[0] - self._seq_len

	@property
	def tokens(self) -> "oa.Matrix":
		"""Full [N] f32 byte-token matrix on the owning engine."""
		return self._tokens

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
			raise IndexError("enwik8 index is outside the dataset")
