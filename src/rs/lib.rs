//! Realm dataset packs and typed loaders for OA.
//!
//! `pack` is the transport layer: immutable manifests, `OA_DATA_DIR` cache
//! resolution, fetch, and SHAKE256-256 verification. Each typed loader —
//! grouped by domain under [`image`] and [`text`] — materializes its pack
//! and implements [`oa::data::Dataset`], so [`oa::data::DataLoader`],
//! [`oa::data::collate`], and subset/split operations work unchanged.

pub mod decode;
mod extract;
mod idx;
mod npy;

pub mod image;
pub mod pack;
pub mod text;
pub mod video;

pub use idx::IdxSplit;
pub use image::{
	Cifar10, Cifar10Config, Cifar10Split, FashionMnist, FashionMnistConfig, FashionMnistSplit,
	KMnist, KMnistConfig, KMnistSplit, Mnist, MnistConfig, MnistSplit,
};
pub use pack::{
	FetchProgress, Pack, PackFile, TerminalProgress, data_root, ensure_pack,
	ensure_pack_with_progress, hash_file,
};
pub use text::{TinyShakespeare, TinyShakespeareConfig};
pub use video::{MovingMnist, MovingMnistConfig};
