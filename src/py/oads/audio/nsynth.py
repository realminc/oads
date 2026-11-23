"""NSynth test split — instrument-family classification corpus.

nsynth-test.jsonwav.tar.gz is materialized through `oads.pack` under
nsynth/1.0.0 and its 4,096 notes extract once to
`<pack>/extracted/wav/` for reuse, with an `index.csv` joining each
note to its family, pitch, velocity, instrument id, and sample count.
The supervised pair is a lazily decoded [1, T] f32 waveform at
16,000 Hz -> the instrument-family index into `families`; pitch,
velocity, and per-instrument ids ride along as [N] f32 matrices.
"""

import csv
from pathlib import Path

import oa

from .. import _native
from .._progress import resolve
from ..dataset import matrix

PACK = _native.pack_manifest("nsynth")

SAMPLE_RATE = 16000

FAMILIES = [
	"bass", "brass", "flute", "guitar", "keyboard", "mallet",
	"organ", "reed", "string", "synth_lead", "vocal",
]


class Nsynth:
	"""Lazy [1, T] WAV waveforms + [N] family labels, 4,096 notes."""

	def __init__(
		self,
		labels: "oa.Matrix",
		pitches: "oa.Matrix",
		velocities: "oa.Matrix",
		instruments: "oa.Matrix",
		lengths: "oa.Matrix",
		notes: list,
		wav_dir: Path,
		engine: "oa.Engine",
	):
		self._labels = labels
		self._pitches = pitches
		self._velocities = velocities
		self._instruments = instruments
		self._lengths = lengths
		self._notes = notes
		self._wav_dir = wav_dir
		self._engine = engine

	@classmethod
	def load(
		cls,
		engine: "oa.Engine | None" = None,
		cache_dir=None,
		progress=True,
	) -> "Nsynth":
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
			notes = [row[0] for row in csv.reader(handle)]
		resolved = engine if engine is not None else oa.Engine.current()
		return cls(
			matrix(fields["labels"], resolved),
			matrix(fields["pitches"], resolved),
			matrix(fields["velocities"], resolved),
			matrix(fields["instruments"], resolved),
			matrix(fields["lengths"], resolved),
			notes,
			extracted / "wav",
			resolved,
		)

	def __len__(self) -> int:
		return len(self._notes)

	@property
	def families(self) -> list:
		"""Instrument-family names; label index `i` names `families[i]`."""
		return FAMILIES

	@property
	def labels(self) -> "oa.Matrix":
		"""[N] f32 instrument-family index matrix into `families`."""
		return self._labels

	@property
	def pitches(self) -> "oa.Matrix":
		"""[N] f32 MIDI pitch values (0-127)."""
		return self._pitches

	@property
	def velocities(self) -> "oa.Matrix":
		"""[N] f32 MIDI velocity values."""
		return self._velocities

	@property
	def instruments(self) -> "oa.Matrix":
		"""[N] f32 upstream instrument ids."""
		return self._instruments

	@property
	def lengths(self) -> "oa.Matrix":
		"""[N] f32 clip sample counts at SAMPLE_RATE."""
		return self._lengths

	def note_id(self, index: int) -> str:
		"""Upstream `<instrument>-<pitch>-<velocity>` id for one row."""
		return self._notes[index]

	def waveform(self, index: int) -> "oa.Matrix":
		"""Lazily decode note `index` to a [1, T] f32 mono waveform."""
		return self.get_item(index)

	def get_item(self, index: int) -> "oa.Matrix":
		"""Lazily decode note `index` to a [1, T] f32 mono waveform."""
		audio = oa.audio.decode_file(
			str(self._wav_dir / f"{self._notes[index]}.wav"),
			self._engine,
		)
		return audio.as_matrix()

	def get_sample(self, index: int):
		"""(waveform, family index) pair for one note."""
		return self.get_item(index), (
			self._labels.slice(0, index, index + 1).reshape([1])
		)
