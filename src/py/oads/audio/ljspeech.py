"""LJSpeech 1.1 single-speaker TTS corpus.

LJSpeech-1.1.tar.bz2 is materialized through `oads.pack` under
ljspeech/1.1.0 and its wavs extract once to `<pack>/extracted/wavs/` for
reuse. The supervised pair is normalized-transcript bytes -> 22,050 Hz
mono PCM: `get_item` returns a [K] padded transcript window and
`get_sample` pairs it with a lazily decoded [1, T] f32 waveform through
`oa.audio.decode_file`. Clip lengths vary, so `oa.data.collate` on
waveforms only batches equal-length clips; `audio_lengths` carries each
clip's sample count for bucketing.
"""

import csv
from pathlib import Path

import oa

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("ljspeech")

SAMPLE_RATE = 22050


class LjSpeech:
	"""[N, K] transcript windows + lazy [1, T] waveforms, 13,100 clips."""

	def __init__(
		self,
		transcripts: "oa.Matrix",
		transcript_lengths: "oa.Matrix",
		audio_lengths: "oa.Matrix",
		ids: list,
		wav_dir: Path,
		engine: "oa.Engine",
	):
		self._transcripts = transcripts
		self._transcript_lengths = transcript_lengths
		self._audio_lengths = audio_lengths
		self._ids = ids
		self._wav_dir = wav_dir
		self._engine = engine

	@classmethod
	def load(
		cls,
		engine: "oa.Engine | None" = None,
		cache_dir=None,
		progress=True,
	) -> "LjSpeech":
		"""Fetch, verify, extract, and upload the transcript surfaces.

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
		with open(extracted / "metadata.csv", newline="") as handle:
			ids = [row[0] for row in csv.reader(handle, delimiter="|")]
		resolved = engine if engine is not None else oa.Engine.current()
		return cls(
			matrix(fields["transcripts"], engine),
			matrix(fields["transcript_lengths"], engine),
			matrix(fields["audio_lengths"], engine),
			ids,
			extracted / "wavs",
			resolved,
		)

	def __len__(self) -> int:
		return len(self._ids)

	@property
	def transcripts(self) -> "oa.Matrix":
		"""[N, K] f32 normalized-transcript bytes, zero-padded."""
		return self._transcripts

	@property
	def transcript_lengths(self) -> "oa.Matrix":
		"""[N] f32 transcript byte lengths."""
		return self._transcript_lengths

	@property
	def audio_lengths(self) -> "oa.Matrix":
		"""[N] f32 waveform sample counts at SAMPLE_RATE."""
		return self._audio_lengths

	def clip_id(self, index: int) -> str:
		"""Clip id (`LJnnn-nnnn`) for one row."""
		return self._ids[index]

	def get_item(self, index: int) -> "oa.Matrix":
		"""[K] padded transcript window for one clip."""
		return (
			self._transcripts.slice(0, index, index + 1)
			.reshape([self._transcripts.shape[1]])
		)

	def waveform(self, index: int) -> "oa.Matrix":
		"""Lazily decode clip `index` to a [1, T] f32 mono waveform."""
		audio = oa.audio.decode_file(
			str(self._wav_dir / f"{self._ids[index]}.wav"), self._engine
		)
		return audio.as_matrix()

	def get_sample(self, index: int):
		"""(transcript window, waveform) pair for one clip."""
		return self.get_item(index), self.waveform(index)
