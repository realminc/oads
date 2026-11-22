"""Tabular datasets: fixed-width numeric feature matrices with class labels.

Every loader uploads a [N, F] f32 feature matrix and a [N] f32 label
matrix, so per-sample slices collate to [B, F] batches through `oa.data`.
Packs arrive unsplit; partition with `oa.data.random_split`.
"""

from .iris import Iris
from .movielens_100k import Movielens100k
from .movielens_1m import Movielens1m
from .titanic import Titanic

__all__ = ["Iris", "Movielens100k", "Movielens1m", "Titanic"]
