"""glue_stsb Python wrapper contract — mirrors the Rust device test."""

from oads.text import GlueStsb


def test_glue_stsb_decodes_split_and_collates(engine):
	dataset = GlueStsb.load(engine=engine, progress=False)
	assert len(dataset) == 5749
	assert dataset.labels.shape == [5749]

	item = dataset.get_item(0)
	assert item.shape[0] == 2
	x, y = dataset.get_sample(5749 - 1)
	assert x.shape[0] == 2
	assert y.shape == [1]
