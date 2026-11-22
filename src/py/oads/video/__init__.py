"""Video datasets: frame-sequence corpora.

Loaders upload a [N, T, C, H, W] f32 frame matrix; per-sample slices
collate to NTCHW batches through `oa.data`.
"""

from .moving_mnist import MovingMnist

__all__ = ["MovingMnist"]
