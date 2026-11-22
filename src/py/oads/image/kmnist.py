"""KMNIST Kuzushiji characters; see `oads.dataset.ImageDataset`."""

from .. import _native
from ..dataset import ImageDataset


class KMnist(ImageDataset):
	PACK = _native.pack_manifest("kmnist")
