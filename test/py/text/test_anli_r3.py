"""anli_r3 Python wrapper contract — mirrors the Rust device test."""

from oads.text import AnliR3


def test_anli_r3_decodes_split(engine):
	dataset = AnliR3.load(engine=engine, progress=False)
	assert len(dataset) == 100459
	assert dataset.labels.shape == [100459]

	item = dataset.get_item(0)
	assert item.shape[0] == 1
	x, y = dataset.get_sample(100459 - 1)
	assert x.shape[0] == 1
	assert y.shape == [1]
