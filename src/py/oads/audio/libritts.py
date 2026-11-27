"""LibriTTS dev-clean multi-speaker TTS corpus.

dev-clean.tar.gz is materialized through `oads.pack` under
libritts/1.0.0 and its 5,736 clips extract once to
`<pack>/extracted/wav/<speaker>/<chapter>/` for reuse, with an
`index.csv` joining each utterance to its speaker and sample count.
The supervised pair is a lazily decoded [1, T] f32 waveform at
24,000 Hz -> the speaker index into `speakers`; normalized transcripts
ride along as a [N, K] padded byte matrix with [N] byte lengths. Clip
lengths vary, so `oa.data.collate` on waveforms only batches
equal-length clips; `lengths` carries each clip's sample count for
bucketing.
"""

import csv
from pathlib import Path

import oa

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("libritts")

SAMPLE_RATE = 24000


class Libritts:
	"""Lazy [1, T] WAV waveforms + [N] speaker ids/lengths, 5,736 clips."""

	def __init__(
		self,
		speaker_ids: "oa.Matrix",
		lengths: "oa.Matrix",
		transcripts: "oa.Matrix",
		transcript_lengths: "oa.Matrix",
		clips: list,
		speakers: list,
		wav_dir: Path,
		engine: "oa.Runtime",
	):
		self._speaker_ids = speaker_ids
		self._lengths = lengths
		self._transcripts = transcripts
		self._transcript_lengths = transcript_lengths
		self._clips = clips
		self._speakers = speakers
		self._wav_dir = wav_dir
		self._engine = engine

	@classmethod
	def load(
		cls,
		engine: "oa.Runtime | None" = None,
		cache_dir=None,
		progress=True,
	) -> "Libritts":
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
		with open(extracted / "speakers.txt") as handle:
			speakers = [line.strip() for line in handle]
		resolved = engine if engine is not None else oa.Runtime.current()
		return cls(
			matrix(fields["speaker_ids"], resolved),
			matrix(fields["lengths"], resolved),
			matrix(fields["transcripts"], resolved),
			matrix(fields["transcript_lengths"], resolved),
			clips,
			speakers,
			extracted / "wav",
			resolved,
		)

	def __len__(self) -> int:
		return len(self._clips)

	@property
	def speakers(self) -> list:
		"""Sorted speaker ids; label index `i` names `speakers[i]`."""
		return self._speakers

	@property
	def speaker_ids(self) -> "oa.Matrix":
		"""[N] f32 speaker index matrix into `speakers`."""
		return self._speaker_ids

	@property
	def lengths(self) -> "oa.Matrix":
		"""[N] f32 clip sample counts at SAMPLE_RATE."""
		return self._lengths

	@property
	def transcripts(self) -> "oa.Matrix":
		"""[N, K] f32 normalized-transcript bytes, zero-padded."""
		return self._transcripts

	@property
	def transcript_lengths(self) -> "oa.Matrix":
		"""[N] f32 transcript byte lengths."""
		return self._transcript_lengths

	def clip_id(self, index: int) -> tuple:
		"""(utterance id, speaker id) for one row."""
		return self._clips[index]

	def waveform(self, index: int) -> "oa.Matrix":
		"""Lazily decode clip `index` to a [1, T] f32 mono waveform."""
		return self.get_item(index)

	def get_item(self, index: int) -> "oa.Matrix":
		"""Lazily decode clip `index` to a [1, T] f32 mono waveform."""
		file, speaker = self._clips[index]
		chapter = file.split("_")[1]
		audio = oa.audio.decode_file(
			str(self._wav_dir / speaker / chapter / f"{file}.wav"),
			self._engine,
		)
		return audio.as_matrix()

	def get_sample(self, index: int):
		"""(waveform, speaker index) pair for one clip."""
		return self.get_item(index), (
			self._speaker_ids.slice(0, index, index + 1).reshape([1])
		)
