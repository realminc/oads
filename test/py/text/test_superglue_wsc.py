"""superglue_wsc Python wrapper contract — mirrors the Rust device test."""

from oads.text import SuperglueWsc


def test_superglue_wsc_decodes_split(engine):
	dataset = SuperglueWsc.load(engine=engine, progress=False)
	assert len(dataset) == 554
	assert dataset.labels.shape == [554]

	item = dataset.get_item(0)
	assert item.shape[0] == 1
	x, y = dataset.get_sample(554 - 1)
	assert x.shape[0] == 1
	assert y.shape == [1]
