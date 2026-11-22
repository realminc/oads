"""Google Speech Commands v0.02; see `oads.dataset.MatrixDataset`.

samples is [N, 1, 16000] f32 mono PCM at 16 kHz — shorter clips are
zero-padded — and labels is [N] f32 class indices into COMMANDS.
"""

from .. import _native
from ..dataset import MatrixDataset


class SpeechCommands(MatrixDataset):
	PACK = _native.pack_manifest("speech_commands")
	FEATURE = "samples"
	COMMANDS = ("backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero")

	@property
	def samples(self):
		"""Full [N, 1, 16000] f32 waveform matrix; alias of `x`."""
		return self.x
