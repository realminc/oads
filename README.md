<p align="center">
  <a href="https://realm.software/">
    <img src="https://dev.realm.software/images/realm/coverData.webp" width="100%" alt="oads — dataset packs for OA">
  </a>
</p>

# oads

**Content-addressed dataset packs and loaders for OA. Manifests in code, payloads in cache, samples on the GPU.**

oads keeps versioned dataset manifests — name, version, license, upstream
mirrors, byte sizes, and `shake256` digests — compiled into the loaders
themselves, the way tfds keeps checksums in code. Payloads are fetched from
upstream into the oads cache on first load, verified, and uploaded to
`oa::Matrix`; they are never committed to git. The Cargo package is
**`oadsrust`**; the library and Python import namespace are **`oads`**.

[![Release](https://img.shields.io/github/v/release/realminc/oads?include_prereleases&label=preview)](https://github.com/realminc/oads/releases)
[![CI](https://github.com/realminc/oads/actions/workflows/ci.yml/badge.svg)](https://github.com/realminc/oads/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/oadsrust?label=crates.io&cacheSeconds=300)](https://crates.io/crates/oadsrust)
[![PyPI](https://img.shields.io/pypi/v/oadspython?label=pypi&cacheSeconds=300)](https://pypi.org/project/oadspython/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[Realm](https://realm.software/) · [OA](https://github.com/realminc/oa) · [Developer guide](https://dev.realm.software/oa)

## Install

```toml
[dependencies]
oadsrust = "0.2"
```

```bash
python -m pip install oadspython
```

Rust code uses `use oads::...`; Python uses `import oads`. Both carry the
same loader contract — fetch, verify, decode, and upload run in Rust
through `oads._native`.

## A few lines of MNIST

```rust
use oa::data::collate;
use oads::{Mnist, MnistConfig};

// Uploads on this thread's default engine (oa::Runtime::current()).
let dataset = Mnist::load(&MnistConfig::default())?;
assert_eq!(dataset.images().shape(), &[60_000, 1, 28, 28]);

let batch = collate(&[dataset.get_sample(0)?, dataset.get_sample(1)?])?;
assert_eq!(batch.x.shape(), &[2, 1, 28, 28]);
# Ok::<(), oa::Error>(())
```

```python
import oads

dataset = oads.Mnist.load()  # engine=None resolves oa.Runtime.current()
assert dataset.images.shape == [60000, 1, 28, 28]

x, y = dataset.get_sample(0)
assert x.shape == [1, 28, 28]
```

Every loader follows the OA engine convention: the engineless `load` /
`load_with_progress` resolve the calling thread's default engine, and the
`*_on` variants — `Mnist::load_on(&engine, &config)`,
`Mnist::load_with_progress_on(&engine, &config, &mut cb)` — pin an explicit
one. Install a default once with `oa::Runtime::set_default` /
`oa.Runtime.set_default` and every subsequent `load` targets it.

Loaders implement `oa::data::Dataset`, so `oa.data` collation, shuffling,
and `random_split` work unchanged — oads produces the data; OA owns the
pipeline, the same way tfds produces `tf.data.Dataset` without living
inside TensorFlow.

## Datasets

A selection of the most-used loaders — the full catalog lives in
[docs/datasets.md](docs/datasets.md):

| Domain | Loader | Samples | Shape | Upstream license |
| --- | --- | --- | --- | --- |
| image | `Mnist` | 60,000 train / 10,000 test | `[N, 1, 28, 28]` F32 | [CC BY-SA 3.0](license/mnist) |
| image | `FashionMnist` | 60,000 train / 10,000 test | `[N, 1, 28, 28]` F32 | [MIT](license/fashion_mnist) |
| image | `Cifar10` | 50,000 train / 10,000 test | `[N, 3, 32, 32]` F32 NCHW | [unspecified](license/cifar10) |
| image | `Cifar100` | 50,000 train / 10,000 test | `[N, 3, 32, 32]` F32 NCHW | [unspecified](license/cifar100) |
| audio | `SpeechCommands` | 84,843 train / 9,981 validation / 11,005 test | `[N, 1, 16000]` F32 PCM | [CC BY 4.0](license/speech_commands) |
| text | `Imdb` | 25,000 train / 25,000 test | `[N, 512]` F32 byte windows | [unspecified](license/imdb) |
| text | `AgNews` | 120,000 train / 7,600 test | `[N, 512]` F32 byte windows | [unspecified](license/ag_news) |
| text | `YelpPolarity` | 560,000 train / 38,000 test | `[N, 512]` F32 byte windows | [unspecified](license/yelp_polarity) |
| text | `TinyShakespeare` | 1,115,394 bytes | `[seq_len]` token windows | [public domain](license/tiny_shakespeare) |
| video | `MovingMnist` | 10,000 sequences | `[T, 1, 64, 64]` per sample | [CC BY-SA 3.0 (mnist derivative)](license/moving_mnist) |

Also in the catalog: `KMnist`, `Emnist` (six variants), `Stl10`,
`MiniSpeechCommands`, `Groove`, `LjSpeech`, `CremaD`, `Vctk`, `Maestro`
(note events, non-commercial license), `Iris`, `Titanic`,
`Movielens100k`, and `Movielens1m`.

Images and labels are F32 — `oa::matrix::slice` is F32-only — with pixels
scaled into `[0, 1]` unless `normalize=false`. `MovingMnist` stores frames
flat as `[N*T, 1, 64, 64]` and returns input/target sequences shifted one
frame apart for next-frame prediction; `TinyShakespeare` returns byte-token
windows and the window shifted one byte ahead. `Maestro` stores one
`[E, 3]` (pitch, step, duration) event matrix and returns fixed
`[seq_len, 3]` device slices paired with the one-event-shifted target —
the next-event prediction contract.

Detection packs (`Voc2007`, `Voc2012`, `Kitti`, `Coco`, `WiderFace`)
trade the resident `[N]` label matrix for a resident `[T, 5]` box
matrix — one `[x1, y1, x2, y2, class]` F32 row per object plus a
`[N + 1]` offsets vector. Images stay on disk and decode lazily;
`get_sample(i)` returns the `[3, H, W]` image and a `[K, 5]` device
slice of the shared box storage. Unlabeled splits (KITTI test, WIDER
test) index normally and slice to `[0, 5]` boxes.

## Pack transport

Every manifest member records its exact byte size, SHAKE256-256 digest,
and mirror list. `ensure_pack` materializes `root/<name>/<version>` under
the cache root — `OA_DATA_DIR`, then `~/.oads`, then the platform cache:

- A member present at the recorded size is a cache hit; nothing is
  rehashed or refetched.
- Downloads write `*.partial` members and resume from the recorded byte
  length (`curl -C -`), so an interrupted fetch continues where it
  stopped. A completed-but-unpublished partial is verified in place.
- Progress flows to a callback — `oads::TerminalProgress` renders the
  bar, percentage, byte counts, speed, elapsed, and ETA in Rust;
  `oads._progress.FetchProgress` renders the same line in Python and is
  the default on `load(progress=...)`.
- A callback abort keeps the partial; the digest gates publication, so a
  corrupt or truncated member never reaches the pack directory.

## Streaming decode

The corpora are larger than workstation memory by design, so decoders
stream archive members instead of materializing them:

- `walk_tar_gz_reader` hands each tar member to the decode callback as a
  bounded `Read`. Members the loader does not select cost no allocation;
  members it does select are consumed line-by-line or through typed
  `serde` records rather than whole-document `Value` trees.
- A `Cancelled` callback error stops the walk early and is swallowed, so
  a `count`-bounded decode does not decompress the rest of the archive.
- Packs whose split materializes GiB-scale F32 (`lm1b`, `xsum`,
  `math_dataset`) expose `count: Option<usize>`; constrained hosts pass
  `Some(n)`.

Measured on an Intel Iris Xe host with 15 GiB RAM, device-test loads
(November 2026):

| Pack | Buffered decode | Streaming decode |
| --- | --- | --- |
| cfq (4.8 GiB member) | member buffer + copy + `Value` tree ≥ 9.5 GiB, OOM-killed | 1.0 GiB peak RSS, 262 s |
| math_dataset (2.3 GiB, ~10M rows) | three archive passes + all-row materialization | single pass, `count` early stop: 8 192 rows in 0.43 s |
| lm1b (1.8 GiB → ~17 GiB F32) | full-blob F32 materialization OOM | `count` early stop: 100 000 sentences in 0.94 s |
| xsum (1.8 GiB, 226k members) | every member buffered before name check | name-filtered member read + `count`: 4 096 documents in 1.21 s |
| trivia_qa (2.5 GiB, 438k members) | every member buffered before name check | zero-allocation skips: 137.65 s |

All loaders follow this contract: archive members stream through the
decoder without full-buffer copies, and `count`-bounded decodes stop
the archive walk early.

## Repository layout

```
src/rs/    oadsrust crate — pack transport and typed loaders by domain
src/py/    oads-python pyo3 crate (oads._native) + thin oads package
test/rs/   host pack tests and Vulkan-gated loader tests, same domains
test/py/   pytest mirrors of the loader contracts
license/   upstream dataset license texts and provenance
```

## Verification

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --test pack                            # host pack tests
cargo test --test image|text|video -- --ignored   # Vulkan device loaders
python3 -m pytest test/py                         # Python mirrors
```

Device tests fetch packs once into the cache and exercise the real upload,
slice, and collate path; host tests cover digest verification, resume,
cancellation, and progress without a device.

## License

The repository is MIT-licensed; see [LICENSE](LICENSE). Each dataset
carries its own upstream license — recorded in the manifest and carried in
[license/](license) — and the loader design is adapted from
tensorflow/datasets (Apache-2.0), credited in the same index.

<p align="center">
  <a href="https://realm.software/">
    <img src="https://raw.githubusercontent.com/realminc/oa/main/sdk/asset/docs/readme/realmIdentityAscii.gif" width="100%" alt="Realm ASCII identity">
  </a>
</p>
