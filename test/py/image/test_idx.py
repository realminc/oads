"""Device-side IDX dataset contract — MNIST, Fashion-MNIST, and KMNIST share
the gzipped IDX format and the [N, 1, 28, 28] f32 image / [N] f32 label
contract. The pack is fetched once into OA_DATA_DIR (or the platform cache)
and reused. Requires a Vulkan device."""

import pytest

import oads

DATASETS = {
	"mnist": oads.Mnist,
	"fashion_mnist": oads.FashionMnist,
	"kmnist": oads.KMnist,
}


@pytest.mark.parametrize("name", DATASETS, ids=DATASETS)
def test_train_split_loads_and_samples(engine, name):
	dataset = DATASETS[name].load(engine, split="train")
	assert len(dataset) == 60_000
	assert dataset.split == "train"
	assert list(dataset.images.shape) == [60_000, 1, 28, 28]
	assert list(dataset.labels.shape) == [60_000]

	labels = dataset.labels.read_f32()
	assert all(v == int(v) and 0 <= v < 10 for v in labels)

	item = dataset.get_item(0)
	assert list(item.shape) == [1, 28, 28]
	x, y = dataset.get_sample(0)
	assert list(x.shape) == [1, 28, 28]
	assert list(y.shape) == [1]

	pixels = dataset.images.slice(0, 0, 256).read_f32()
	assert all(0.0 <= v <= 1.0 for v in pixels)


@pytest.mark.parametrize("name", DATASETS, ids=DATASETS)
def test_test_split_has_expected_shape(engine, name):
	dataset = DATASETS[name].load(engine, split="test")
	assert len(dataset) == 10_000
	assert list(dataset.images.shape) == [10_000, 1, 28, 28]
