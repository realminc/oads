"""CIFAR-10 color objects; see `oads.dataset.ImageDataset`."""

from .. import _native
from ..dataset import ImageDataset


class Cifar10(ImageDataset):
	PACK = _native.pack_manifest("cifar10")
