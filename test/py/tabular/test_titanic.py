"""Device-side Titanic dataset contract."""

import oads


def test_titanic_loads_and_collates(engine):
	dataset = oads.Titanic.load(engine)
	assert len(dataset) == 891
	assert list(dataset.features.shape) == [891, 9]
	labels = dataset.labels.read_f32()
	assert all(v in (0.0, 1.0) for v in labels)
	features = dataset.features.read_f32()
	assert all(0.0 <= v <= 1.0 for v in features)
	x, y = dataset.get_sample(0)
	assert list(x.shape) == [9]
