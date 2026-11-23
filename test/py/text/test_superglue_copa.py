"""superglue_copa Python wrapper contract — mirrors the Rust device test."""

from oads.text import SuperglueCopa


def test_superglue_copa_decodes_split(engine):
	dataset = SuperglueCopa.load(engine=engine, progress=False)
	assert len(dataset) == 400
	assert dataset.labels.shape == [400]

	item = dataset.get_item(0)
	assert item.shape[0] == 1
	x, y = dataset.get_sample(400 - 1)
	assert x.shape[0] == 1
	assert y.shape == [1]
