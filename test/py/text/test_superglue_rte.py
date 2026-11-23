"""superglue_rte Python wrapper contract — mirrors the Rust device test."""

from oads.text import SuperglueRte


def test_superglue_rte_decodes_split(engine):
	dataset = SuperglueRte.load(engine=engine, progress=False)
	assert len(dataset) == 2490
	assert dataset.labels.shape == [2490]

	item = dataset.get_item(0)
	assert item.shape[0] == 1
	x, y = dataset.get_sample(2490 - 1)
	assert x.shape[0] == 1
	assert y.shape == [1]
