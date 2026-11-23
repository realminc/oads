"""glue_sst2 Python wrapper contract — mirrors the Rust device test."""

from oads.text import GlueSst2


def test_glue_sst2_decodes_split_and_collates(engine):
	dataset = GlueSst2.load(engine=engine, progress=False)
	assert len(dataset) == 67349
	assert dataset.labels.shape == [67349]

	item = dataset.get_item(0)
	assert item.shape[0] == 1
	x, y = dataset.get_sample(67349 - 1)
	assert x.shape[0] == 1
	assert y.shape == [1]
