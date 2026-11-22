"""Google mini Speech Commands; see `oads.dataset.MatrixDataset`.

samples is [N, 1, 16000] f32 mono PCM at 16 kHz — shorter clips are
zero-padded — and labels is [N] f32 class indices into COMMANDS. The pack
has no official split; shuffle and partition through oa.data.
"""

from pathlib import Path

from .. import _native
from .._progress import resolve
from ..dataset import MatrixDataset, matrix


class MiniSpeechCommands(MatrixDataset):
	PACK = _native.pack_manifest("mini_speech_commands")
	FEATURE = "samples"
	COMMANDS = ("down", "go", "left", "no", "right", "stop", "up", "yes")

	@classmethod
	def load(cls, engine=None, cache_dir=None, normalize: bool = True, progress=True):
		"""Fetch, verify, decode, and upload all 8,000 command clips.

		normalize=True scales PCM into [-1, 1]; normalize=False keeps raw
		i16 amplitudes as f32. progress=True renders the download bar.
		"""
		fields = _native.load(
			cls.PACK["name"],
			# The pack carries one unsplit set; the native split argument
			# only accepts train/test and is ignored for this loader.
			split="train",
			normalize=normalize,
			cache_dir=str(Path(cache_dir)) if cache_dir else None,
			progress=resolve(progress),
		)
		return cls(
			matrix(fields[cls.FEATURE], engine),
			matrix(fields[cls.LABEL], engine),
			"all",
		)

	@property
	def samples(self):
		"""Full [N, 1, 16000] f32 waveform matrix; alias of `x`."""
		return self.x
