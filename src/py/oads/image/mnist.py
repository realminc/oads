"""MNIST handwritten digits; see `oads.dataset.ImageDataset`."""

from .. import _native
from ..dataset import ImageDataset


class Mnist(ImageDataset):
	PACK = _native.pack_manifest("mnist")
