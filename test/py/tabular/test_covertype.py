"""covertype Python wrapper contract — mirrors the Rust device test."""

from oads.tabular import Covertype


def test_covertype_loads(engine):
	dataset = Covertype.load(engine=engine, progress=False)
	assert len(dataset) == 581_012
	assert dataset.x.shape == [581_012, 54]
	assert dataset.labels.shape == [581_012]

	x, y = dataset.get_sample(0)
	assert x.shape == [54]
