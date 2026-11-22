"""oads — Realm dataset packs and typed loaders for OA.

pack is the transport layer: manifests, OA_DATA_DIR cache resolution, fetch,
and SHAKE256-256 verification. Typed loaders, grouped under audio, image,
text, and video, materialize their pack and expose an oa.data-compatible
sample contract. Fetch and decode run in Rust through the compiled
`oads._native` extension; the Python layer owns dataset semantics and
matrix upload.
"""

from . import audio, dataset, image, pack, tabular, text, video
from .audio import CremaD, Groove, LjSpeech, Maestro, MiniSpeechCommands, SpeechCommands, Vctk
from .image import Cifar10, Cifar100, Emnist, FashionMnist, KMnist, Mnist, Stl10
from .tabular import Iris, Movielens100k, Movielens1m, Titanic
from .text import AgNews, Imdb, TinyShakespeare, YelpPolarity
from .video import MovingMnist

__all__ = [
	"AgNews",
	"Cifar10",
	"Cifar100",
	"CremaD",
	"Emnist",
	"FashionMnist",
	"Groove",
	"Imdb",
	"Iris",
	"KMnist",
	"LjSpeech",
	"Maestro",
	"MiniSpeechCommands",
	"Mnist",
	"Movielens100k",
	"Movielens1m",
	"MovingMnist",
	"SpeechCommands",
	"Stl10",
	"TinyShakespeare",
	"Titanic",
	"Vctk",
	"YelpPolarity",
	"audio",
	"dataset",
	"image",
	"pack",
	"tabular",
	"text",
	"video",
]
__version__ = "0.2.0"
