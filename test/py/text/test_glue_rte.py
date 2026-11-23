"""glue_rte Python wrapper contract — mirrors the Rust device test."""

from oads.text import GlueRte


def test_glue_rte_decodes_split_and_collates(engine):
	dataset = GlueRte.load(engine=engine, progress=False)
	assert len(dataset) == 2490
	assert dataset.labels.shape == [2490]

	item = dataset.get_item(0)
	assert item.shape[0] == 2
	x, y = dataset.get_sample(2490 - 1)
	assert x.shape[0] == 2
	assert y.shape == [1]
