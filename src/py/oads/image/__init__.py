"""Image datasets: IDX-digit packs and archive-decoded photographic corpora.

Every loader uploads a [N, C, H, W] f32 image matrix and a [N] f32 label
matrix, so per-sample slices collate to NCHW batches through `oa.data`.
"""

from .beans import Beans
from .cars196 import Cars196
from .cats_vs_dogs import CatsVsDogs
from .cifar10 import Cifar10
from .cifar100 import Cifar100
from .coco import Coco
from .coco_captions import CocoCaptions
from .cub200 import Cub200
from .dtd import Dtd
from .emnist import Emnist
from .eurosat import Eurosat
from .fashion_mnist import FashionMnist
from .food101 import Food101
from .horses_or_humans import HorsesOrHumans
from .imagenette import Imagenette
from .imagewang import Imagewang
from .kmnist import KMnist
from .kitti import Kitti
from .lfw import Lfw
from .mnist import Mnist
from .omniglot import Omniglot
from .oxford_flowers102 import OxfordFlowers102
from .oxford_iiit_pet import OxfordIiitPet
from .patch_camelyon import PatchCamelyon
from .places365_small import Places365Small
from .quickdraw import Quickdraw
from .rock_paper_scissors import RockPaperScissors
from .smallnorb import Smallnorb
from .stanford_dogs import StanfordDogs
from .stl10 import Stl10
from .svhn import Svhn
from .uc_merced import UcMerced
from .voc2007 import Voc2007
from .voc2012 import Voc2012
from .wider_face import WiderFace

__all__ = ["Beans", "Cars196", "CatsVsDogs", "Cifar10", "Cifar100", "Coco", "CocoCaptions", "Cub200", "Dtd", "Emnist", "Eurosat", "FashionMnist", "Food101", "HorsesOrHumans", "Imagenette", "Imagewang", "KMnist", "Kitti", "Lfw", "Mnist", "Omniglot", "OxfordFlowers102", "OxfordIiitPet", "PatchCamelyon", "Places365Small", "Quickdraw", "RockPaperScissors", "Smallnorb", "StanfordDogs", "Stl10", "Svhn", "UcMerced", "Voc2007", "Voc2012", "WiderFace"]
