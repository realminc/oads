"""FLORES-101 many-to-many translation evaluation sentences.

The pack is materialized through `oads.pack` under `flores101/1.0.0`;
the selected member set pairs every non-English `dev`/`devtest` file
with English into `english`/`target` `[N, K_i]` f32 byte matrices and a
`labels` language index (sorted member order, `classes[i]` is the ISO
code). `get_item` returns the `[1, sum(K_i)]` end-to-end field row.
"""

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("flores101")
FIELDS = ['english', 'target']
CLASSES = ['afr', 'amh', 'ara', 'asm', 'ast', 'azj', 'bel', 'ben', 'bos', 'bul', 'cat', 'ceb', 'ces', 'ckb', 'cym', 'dan', 'deu', 'ell', 'est', 'fas', 'fin', 'fra', 'ful', 'gle', 'glg', 'guj', 'hau', 'heb', 'hin', 'hrv', 'hun', 'hye', 'ibo', 'ind', 'isl', 'ita', 'jav', 'jpn', 'kam', 'kan', 'kat', 'kaz', 'kea', 'khm', 'kir', 'kor', 'lao', 'lav', 'lin', 'lit', 'ltz', 'lug', 'luo', 'mal', 'mar', 'mkd', 'mlt', 'mon', 'mri', 'msa', 'mya', 'nld', 'nob', 'npi', 'nso', 'nya', 'oci', 'orm', 'ory', 'pan', 'pol', 'por', 'pus', 'ron', 'rus', 'slk', 'slv', 'sna', 'snd', 'som', 'spa', 'srp', 'swe', 'swh', 'tam', 'tel', 'tgk', 'tgl', 'tha', 'tur', 'ukr', 'umb', 'urd', 'uzb', 'vie', 'wol', 'xho', 'yor', 'zho_simpl', 'zho_trad', 'zul']

class Flores101:
	"""Flores101 2-field text matrices plus `labels`."""

	def __init__(self, fields, labels, classes):
		self._fields = fields
		self._labels = labels
		self._classes = classes

	@classmethod
	def load(
		cls,
		engine: "oa.Runtime | None" = None,
		split: str = "train",
		cache_dir=None,
		progress=True,
	) -> "Flores101":
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
	def english(self) -> "oa.Matrix":
		"""[N, K] f32 `english` field, zero-padded."""
		return self._fields[0]

	@property
	def target(self) -> "oa.Matrix":
		"""[N, K] f32 `target` field, zero-padded."""
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
