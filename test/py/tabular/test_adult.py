"""adult Python wrapper contract — mirrors the Rust device test."""

from oads.tabular import Adult


def test_adult_loads_test_split(engine):
	dataset = Adult.load(engine=engine, split="test", progress=False)
	assert len(dataset) == 16_281
	assert dataset.x.shape == [16_281, 92]
	assert dataset.labels.shape == [16_281]

	x, y = dataset.get_sample(0)
	assert x.shape == [92]
