"""glue_cola Python wrapper contract — mirrors the Rust device test."""

from oads.text import GlueCola


def test_glue_cola_decodes_split_and_collates(engine):
	dataset = GlueCola.load(engine=engine, progress=False)
	assert len(dataset) == 8551
	assert dataset.labels.shape == [8551]

	item = dataset.get_item(0)
	assert item.shape[0] == 1
	x, y = dataset.get_sample(8551 - 1)
	assert x.shape[0] == 1
	assert y.shape == [1]
