"""oads — Realm dataset packs and typed loaders for OA.

pack is the transport layer: manifests, OA_DATA_DIR cache resolution, fetch,
and SHAKE256-256 verification. Typed loaders, grouped under image, text, and
video, materialize their pack and expose an oa.data-compatible sample
contract. Fetch and decode run in Rust through the compiled `oads._native`
extension; the Python layer owns dataset semantics and matrix upload.
"""

from . import dataset, image, pack, text, video
from .image import Cifar10, FashionMnist, KMnist, Mnist
from .text import TinyShakespeare
from .video import MovingMnist

__all__ = [
	"Cifar10",
	"FashionMnist",
	"KMnist",
	"Mnist",
	"MovingMnist",
	"TinyShakespeare",
	"image",
	"dataset",
	"pack",
	"text",
	"video",
]
__version__ = "0.1.0"
