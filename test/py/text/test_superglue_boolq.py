"""superglue_boolq Python wrapper contract — mirrors the Rust device test."""

from oads.text import SuperglueBoolq


def test_superglue_boolq_decodes_split(engine):
	dataset = SuperglueBoolq.load(engine=engine, progress=False)
	assert len(dataset) == 9427
	assert dataset.labels.shape == [9427]

	item = dataset.get_item(0)
	assert item.shape[0] == 1
	x, y = dataset.get_sample(9427 - 1)
	assert x.shape[0] == 1
	assert y.shape == [1]
