"""Device-side AG News dataset contract."""

import oads


def test_test_split_loads_and_samples(engine):
	dataset = oads.AgNews.load(engine, split="test")
	assert len(dataset) == 7_600
	assert list(dataset.text.shape) == [7_600, 512]
	labels = dataset.labels.read_f32()
	assert all(v == int(v) and 0 <= v < 4 for v in labels)
	x, y = dataset.get_sample(0)
	assert list(x.shape) == [512]
	assert list(y.shape) == [1]
