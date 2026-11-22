"""Device-side Yelp Polarity dataset contract."""

import oads


def test_test_split_loads_and_samples(engine):
	dataset = oads.YelpPolarity.load(engine, split="test")
	assert len(dataset) == 38_000
	assert list(dataset.text.shape) == [38_000, 512]
	labels = dataset.labels.read_f32()
	assert all(v in (0.0, 1.0) for v in labels)
	x, y = dataset.get_sample(0)
	assert list(x.shape) == [512]
