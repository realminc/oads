"""forest_fires Python wrapper contract — mirrors the Rust device test."""

from oads.tabular import ForestFires


def test_forest_fires_decodes_split(engine):
	dataset = ForestFires.load(engine=engine, progress=False)
	assert len(dataset) == 517
	assert list(dataset.features.shape) == [517, 12]
	assert list(dataset.labels.shape) == [517]
	x, y = dataset.get_sample(0)
	assert list(x.shape) == [12]
