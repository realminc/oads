"""superglue_multirc Python wrapper contract — mirrors the Rust device test."""

from oads.text import SuperglueMultirc


def test_superglue_multirc_decodes_split(engine):
	dataset = SuperglueMultirc.load(engine=engine, progress=False)
	assert len(dataset) == 27243
	assert dataset.labels.shape == [27243]

	item = dataset.get_item(0)
	assert item.shape[0] == 1
	x, y = dataset.get_sample(27243 - 1)
	assert x.shape[0] == 1
	assert y.shape == [1]
