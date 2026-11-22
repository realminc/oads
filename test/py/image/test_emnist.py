"""Device-side EMNIST dataset contract (balanced variant)."""

import oads


def test_train_split_loads_and_samples(engine):
	dataset = oads.Emnist.load(engine, split="train")
	assert len(dataset) == 112_800
	assert list(dataset.images.shape) == [112_800, 1, 28, 28]
	labels = dataset.labels.read_f32()
	assert all(v == int(v) and 0 <= v < 47 for v in labels)
	x, y = dataset.get_sample(0)
	assert list(x.shape) == [1, 28, 28]
	assert list(y.shape) == [1]
	pixels = dataset.images.slice(0, 0, 256).read_f32()
	assert all(0.0 <= v <= 1.0 for v in pixels)


def test_test_split_has_expected_shape(engine):
	dataset = oads.Emnist.load(engine, split="test")
	assert len(dataset) == 18_800
	assert list(dataset.images.shape) == [18_800, 1, 28, 28]
