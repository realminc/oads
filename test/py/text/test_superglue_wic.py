"""superglue_wic Python wrapper contract — mirrors the Rust device test."""

from oads.text import SuperglueWic


def test_superglue_wic_decodes_split(engine):
	dataset = SuperglueWic.load(engine=engine, progress=False)
	assert len(dataset) == 5428
	assert dataset.labels.shape == [5428]

	item = dataset.get_item(0)
	assert item.shape[0] == 1
	x, y = dataset.get_sample(5428 - 1)
	assert x.shape[0] == 1
	assert y.shape == [1]
