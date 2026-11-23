"""superglue_cb Python wrapper contract — mirrors the Rust device test."""

from oads.text import SuperglueCb


def test_superglue_cb_decodes_split(engine):
	dataset = SuperglueCb.load(engine=engine, progress=False)
	assert len(dataset) == 250
	assert dataset.labels.shape == [250]

	item = dataset.get_item(0)
	assert item.shape[0] == 1
	x, y = dataset.get_sample(250 - 1)
	assert x.shape[0] == 1
	assert y.shape == [1]
