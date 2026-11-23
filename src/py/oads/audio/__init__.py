"""Audio datasets: clip archives decoded to PCM waveform matrices and MIDI
performances decoded to (pitch, step, duration) event matrices.

Waveform loaders upload a [N, 1, S] f32 sample matrix and a [N] f32 label
matrix; sequence loaders expose fixed-length event windows as device
slices. Both collate through `oa.data` without padding.
"""

from .crema_d import CremaD
from .groove import Groove
from .gtzan import Gtzan
from .librispeech import Librispeech
from .libritts import Libritts
from .ljspeech import LjSpeech
from .maestro import Maestro
from .mini_speech_commands import MiniSpeechCommands
from .nsynth import Nsynth
from .speech_commands import SpeechCommands
from .vctk import Vctk
from .waves_yesno import WavesYesno

__all__ = ["CremaD", "Groove", "Gtzan", "Librispeech", "Libritts", "LjSpeech", "Maestro", "MiniSpeechCommands", "Nsynth", "SpeechCommands", "Vctk", "WavesYesno"]
