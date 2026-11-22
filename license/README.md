# Licenses

oads source is MIT ([LICENSE](../LICENSE)). The loading model is adapted
from tensorflow/datasets (Apache-2.0); its text lives in
`tensorflow_datasets/` per the attribution table below. The payloads
fetched into the oads cache are upstream datasets under their own
licenses; this repository ships loaders and manifests, not the data
itself.

| Pack | Upstream | License | Text |
| --- | --- | --- | --- |
| `mnist` | Yann LeCun et al., cvdf-datasets / ossci-datasets | CC BY-SA 3.0 | [mnist/CC-BY-SA-3.0.txt](mnist/CC-BY-SA-3.0.txt) |
| `fashion_mnist` | Zalando SE | MIT | [fashion_mnist/MIT.txt](fashion_mnist/MIT.txt) |
| `tiny_shakespeare` | karpathy/char-rnn via Project Gutenberg | public domain | [tiny_shakespeare/public-domain.txt](tiny_shakespeare/public-domain.txt) |
| `kmnist` | ROIS-CDS, National Institute of Informatics | CC BY-SA 4.0 | [kmnist/CC-BY-SA-4.0.txt](kmnist/CC-BY-SA-4.0.txt) |
| `cifar10` | University of Toronto (Krizhevsky, Hinton) | unspecified | [cifar10/provenance.txt](cifar10/provenance.txt) |
| `moving_mnist` | University of Toronto (Srivastava et al.) | CC BY-SA 3.0 (mnist derivative) | [moving_mnist/provenance.txt](moving_mnist/provenance.txt) |

The `tensorflow_datasets/` entry below is code attribution, not a dataset pack.

| Material | Source | License | Text |
| --- | --- | --- | --- |
| tensorflow/datasets design | github.com/tensorflow/datasets | Apache-2.0 | [tensorflow_datasets/Apache-2.0.txt](tensorflow_datasets/Apache-2.0.txt) |

CC BY-SA 3.0 requires attribution and share-alike redistribution of the data
or derivatives; keep notices with any republished corpus. MIT requires the
notice to ride along with copies. Each pack manifest carries its license
identifier, so loaders can report it programmatically.
