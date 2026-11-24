"""Device-side PatchCamelyon dataset contract."""

import oads


def test_validation_split_loads_and_samples(engine):
	dataset = oads.PatchCamelyon.load(engine, split="validation", count=1000)
	assert len(dataset) == 1000
	assert list(dataset.images.shape) == [1000, 3, 96, 96]
	labels = dataset.labels.read_f32()
	assert all(v == int(v) and 0 <= v < 2 for v in labels)
	x, y = dataset.get_sample(0)
	assert list(x.shape) == [3, 96, 96]
	assert list(y.shape) == [1]
