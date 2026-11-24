"""dbpedia Python wrapper contract — mirrors the Rust device test."""

from oads.text import Dbpedia


def test_dbpedia_loads_test_split(engine):
	dataset = Dbpedia.load(engine=engine, split="test", progress=False)
	assert len(dataset) == 70_000
	assert dataset.labels.shape == [70_000]

	x, y = dataset.get_sample(0)
	assert y.shape == [1]
