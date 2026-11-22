"""Fashion-MNIST clothing items; see `oads.dataset.ImageDataset`."""

from .. import _native
from ..dataset import ImageDataset


class FashionMnist(ImageDataset):
	PACK = _native.pack_manifest("fashion_mnist")
