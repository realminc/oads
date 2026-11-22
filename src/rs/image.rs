//! Image datasets: IDX-digit packs and archive-decoded photographic corpora.
//!
//! Every loader uploads a `[N, C, H, W]` F32 image tensor and a `[N]` F32
//! label tensor, so per-sample slices collate to NCHW batches through
//! [`oa::data`].

pub mod cifar10;
pub mod fashion_mnist;
pub mod kmnist;
pub mod mnist;

pub use cifar10::{Cifar10, Cifar10Config, Cifar10Split};
pub use fashion_mnist::{FashionMnist, FashionMnistConfig, FashionMnistSplit};
pub use kmnist::{KMnist, KMnistConfig, KMnistSplit};
pub use mnist::{Mnist, MnistConfig, MnistSplit};
