"""Device-side CIFAR-10 dataset contract."""

import oads


def test_train_split_loads_and_samples(engine):
	dataset = oads.Cifar10.load(engine, split="train")
	assert len(dataset) == 50_000
	assert list(dataset.images.shape) == [50_000, 3, 32, 32]
	labels = dataset.labels.read_f32()
	assert all(v == int(v) and 0 <= v < 10 for v in labels)
	x, y = dataset.get_sample(0)
	assert list(x.shape) == [3, 32, 32]
	assert list(y.shape) == [1]
	pixels = dataset.images.slice(0, 0, 256).read_f32()
	assert all(0.0 <= v <= 1.0 for v in pixels)


def test_test_split_has_expected_shape(engine):
	dataset = oads.Cifar10.load(engine, split="test")
	assert len(dataset) == 10_000
	assert list(dataset.images.shape) == [10_000, 3, 32, 32]
