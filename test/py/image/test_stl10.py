"""Device-side STL-10 dataset contract."""

import oads


def test_test_split_loads_and_samples(engine):
	dataset = oads.Stl10.load(engine, split="test")
	assert len(dataset) == 8_000
	assert list(dataset.images.shape) == [8_000, 3, 96, 96]
	labels = dataset.labels.read_f32()
	assert all(v == int(v) and 0 <= v <= 9 for v in labels)
	x, y = dataset.get_sample(0)
	assert list(x.shape) == [3, 96, 96]
