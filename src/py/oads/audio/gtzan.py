"""GTZAN genre collection (marsyas mirror).

genres.tar.gz is materialized through `oads.pack` under `gtzan/1.0.0`
and its 999 decodable clips extract once to `<pack>/extracted/wav/<genre>/` for
reuse, with an `index.csv` joining each clip to its genre and sample
count. The supervised pair is a lazily decoded [1, T] f32 WAV waveform
at 22,050 Hz -> the genre index into `genres`. Clip lengths vary, so `oa.data.collate` on waveforms only batches
equal-length clips; `lengths` carries each clip's sample count for
bucketing.
"""

import csv
from pathlib import Path

import oa

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("gtzan")

SAMPLE_RATE = 22050


class Gtzan:
	"""Lazy [1, T] WAV waveforms + [N] genre ids/lengths, 999 clips
	(the corrupt upstream `jazz.00054.wav` member is skipped)."""

	def __init__(
		self,
		genre_ids: "oa.Matrix",
		lengths: "oa.Matrix",
		clips: list,
		genres: list,
		wav_dir: Path,
		engine: "oa.Engine",
	):
		self._genre_ids = genre_ids
		self._lengths = lengths
		self._clips = clips
		self._genres = genres
		self._wav_dir = wav_dir
		self._engine = engine

	@classmethod
	def load(
		cls,
		engine: "oa.Engine | None" = None,
		cache_dir=None,
		progress=True,
	) -> "Gtzan":
		"""Fetch, verify, extract, and index the pack.

		Waveforms stay in the extracted cache and decode lazily per
		sample. progress=True renders the download bar; a callable
		receives (file, fetched, total) samples instead.
		"""
		fields = _native.load(
			PACK["name"],
			split="train",
			cache_dir=str(Path(cache_dir)) if cache_dir else None,
			progress=resolve(progress),
		)
		root = Path(cache_dir) if cache_dir else Path(_native.data_root())
		extracted = root / PACK["name"] / PACK["version"] / "extracted"
		with open(extracted / "index.csv", newline="") as handle:
			clips = [(row[0], row[1]) for row in csv.reader(handle)]
		with open(extracted / "genres.txt") as handle:
			genres = [line.strip() for line in handle]
		resolved = engine if engine is not None else oa.Engine.current()
		return cls(
			matrix(fields["genre_ids"], resolved),
			matrix(fields["lengths"], resolved),
			clips,
			genres,
			extracted / "wav",
			resolved,
		)

	def __len__(self) -> int:
		return len(self._clips)

	@property
	def genres(self) -> list:
		"""Sorted genre names; label index `i` names `genres[i]`."""
		return self._genres

	@property
	def genre_ids(self) -> "oa.Matrix":
		"""[N] f32 genre index matrix into `genres`."""
		return self._genre_ids

	@property
	def lengths(self) -> "oa.Matrix":
		"""[N] f32 clip sample counts at SAMPLE_RATE."""
		return self._lengths

	def clip_id(self, index: int) -> tuple:
		"""(clip stem, genre) for one row."""
		return self._clips[index]

	def waveform(self, index: int) -> "oa.Matrix":
		"""Lazily decode clip `index` to a [1, T] f32 mono waveform."""
		return self.get_item(index)

	def get_item(self, index: int) -> "oa.Matrix":
		"""Lazily decode clip `index` to a [1, T] f32 mono waveform."""
		file, genre = self._clips[index]
		audio = oa.audio.decode_file(
			str(self._wav_dir / genre / f"{file}.wav"),
			self._engine,
		)
		return audio.as_matrix()

	def get_sample(self, index: int):
		"""(waveform, genre index) pair for one clip."""
		return self.get_item(index), (
			self._genre_ids.slice(0, index, index + 1).reshape([1])
		)
