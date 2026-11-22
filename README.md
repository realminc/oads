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

// Uploads on this thread's default engine (oa::Engine::current()).
let dataset = Mnist::load(&MnistConfig::default())?;
assert_eq!(dataset.images().shape(), &[60_000, 1, 28, 28]);

let batch = collate(&[dataset.get_sample(0)?, dataset.get_sample(1)?])?;
assert_eq!(batch.x.shape(), &[2, 1, 28, 28]);
# Ok::<(), oa::Error>(())
```

```python
import oads

dataset = oads.Mnist.load()  # engine=None resolves oa.Engine.current()
assert dataset.images.shape == [60000, 1, 28, 28]

x, y = dataset.get_sample(0)
assert x.shape == [1, 28, 28]
```

Every loader follows the OA engine convention: the engineless `load` /
`load_with_progress` resolve the calling thread's default engine, and the
`*_on` variants — `Mnist::load_on(&engine, &config)`,
`Mnist::load_with_progress_on(&engine, &config, &mut cb)` — pin an explicit
one. Install a default once with `oa::Engine::set_default` /
`oa.Engine.set_default` and every subsequent `load` targets it.

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
