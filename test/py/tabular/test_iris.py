"""Device-side Iris dataset contract."""

import oads


def test_iris_loads_and_collates(engine):
	dataset = oads.Iris.load(engine)
	assert len(dataset) == 150
	assert list(dataset.features.shape) == [150, 4]
	labels = dataset.labels.read_f32()
	assert all(v == int(v) and 0 <= v <= 2 for v in labels)
	features = dataset.features.read_f32()
	assert all(0.0 <= v <= 1.0 for v in features)
	assert len(oads.Iris.CLASSES) == 3
	x, y = dataset.get_sample(0)
	assert list(x.shape) == [4]
