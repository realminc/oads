"""VCTK 0.92 multi-speaker English speech corpus.

DS_10283_3443.zip is materialized through `oads.pack` under vctk/0.92.0
and its `wav48_silence_trimmed/<speaker>/*_mic{1,2}.flac` members
extract once to `<pack>/extracted/flac/` for reuse (the archive's
nested VCTK-Corpus-0.92.zip member is a redundant copy and is
skipped). The supervised pair is a lazily decoded [1, T] f32 FLAC
waveform at 48,000 Hz -> the speaker index into `speakers`; `txt/`
transcripts ride along as a [N, K] padded byte matrix with [N] byte
lengths. Each microphone channel has slightly different utterance
coverage, so `mic` selects the index; the corpus ships no official
split — partition with `oa.data.random_split`, or group by speaker
for speaker-disjoint evaluation.
"""

import csv
from pathlib import Path

import oa

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("vctk")

SAMPLE_RATE = 48000


class Vctk:
	"""Lazy [1, T] FLAC waveforms + [N] speaker ids/lengths."""

	def __init__(
		self,
		speaker_ids: "oa.Matrix",
		lengths: "oa.Matrix",
		transcripts: "oa.Matrix",
		transcript_lengths: "oa.Matrix",
		clips: list,
		speakers: list,
		flac_dir: Path,
		mic: str,
		engine: "oa.Engine",
	):
		self._speaker_ids = speaker_ids
		self._lengths = lengths
		self._transcripts = transcripts
		self._transcript_lengths = transcript_lengths
		self._clips = clips
		self._speakers = speakers
		self._flac_dir = flac_dir
		self._mic = mic
		self._engine = engine

	@classmethod
	def load(
		cls,
		engine: "oa.Engine | None" = None,
		mic: str = "mic1",
		cache_dir=None,
		progress=True,
	) -> "Vctk":
		"""Fetch, verify, extract, and upload the label surfaces.

		mic is 'mic1' (head-mounted) or 'mic2' (desk). Waveforms stay
		in the extracted cache and decode lazily per sample.
		progress=True renders the download bar; a callable receives
		(file, fetched, total) samples instead.
		"""
		if mic not in ("mic1", "mic2"):
			raise ValueError(f"vctk mic must be 'mic1' or 'mic2', got {mic!r}")
		fields = _native.load(
			PACK["name"],
			split="train",
			cache_dir=str(Path(cache_dir)) if cache_dir else None,
			progress=resolve(progress),
		)
		suffix = "" if mic == "mic1" else "_mic2"
		root = Path(cache_dir) if cache_dir else Path(_native.data_root())
		extracted = root / PACK["name"] / PACK["version"] / "extracted"
		with open(extracted / f"index_{mic}.csv", newline="") as handle:
			clips = [(row[0], row[1]) for row in csv.reader(handle)]
		speakers = (extracted / "speakers.txt").read_text().splitlines()
		resolved = engine if engine is not None else oa.Engine.current()
		return cls(
			matrix(fields[f"speaker_ids{suffix}"], engine),
			matrix(fields[f"lengths{suffix}"], engine),
			matrix(fields[f"transcripts{suffix}"], engine),
			matrix(fields[f"transcript_lengths{suffix}"], engine),
			clips,
			speakers,
			extracted / "flac",
			mic,
			resolved,
		)

	def __len__(self) -> int:
		return len(self._clips)

	@property
	def speaker_ids(self) -> "oa.Matrix":
		"""[N] f32 speaker indices into `speakers`."""
		return self._speaker_ids

	@property
	def lengths(self) -> "oa.Matrix":
		"""[N] f32 waveform sample counts at SAMPLE_RATE."""
		return self._lengths

	@property
	def transcripts(self) -> "oa.Matrix":
		"""[N, K] f32 transcript bytes, zero-padded."""
		return self._transcripts

	@property
	def transcript_lengths(self) -> "oa.Matrix":
		"""[N] f32 transcript byte lengths."""
		return self._transcript_lengths

	@property
	def speakers(self) -> list:
		"""Sorted `pNNN` speaker ids; label index `i` names `speakers[i]`."""
		return self._speakers

	@property
	def mic(self) -> str:
		return self._mic

	def clip_id(self, index: int):
		"""(file id, speaker id) for one row."""
		return self._clips[index]

	def waveform(self, index: int) -> "oa.Matrix":
		"""Lazily decode clip `index` to a [1, T] f32 mono waveform."""
		return self.get_item(index)

	def get_item(self, index: int) -> "oa.Matrix":
		"""Lazily decode clip `index` to a [1, T] f32 mono waveform."""
		file_id, speaker = self._clips[index]
		audio = oa.audio.decode_file(
			str(self._flac_dir / speaker / f"{file_id}_{self._mic}.flac"),
			self._engine,
		)
		return audio.as_matrix()

	def get_sample(self, index: int):
		"""(waveform, speaker label) pair for one clip."""
		return self.get_item(index), self._speaker_ids.slice(
			0, index, index + 1
		)
