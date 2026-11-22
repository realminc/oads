"""Image datasets: IDX-digit packs and archive-decoded photographic corpora.

Every loader uploads a [N, C, H, W] f32 image matrix and a [N] f32 label
matrix, so per-sample slices collate to NCHW batches through `oa.data`.
"""

from .cifar10 import Cifar10
from .cifar100 import Cifar100
from .emnist import Emnist
from .fashion_mnist import FashionMnist
from .kmnist import KMnist
from .mnist import Mnist
from .stl10 import Stl10

__all__ = ["Cifar10", "Cifar100", "Emnist", "FashionMnist", "KMnist", "Mnist", "Stl10"]
