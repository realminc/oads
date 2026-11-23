"""anli_r1 Python wrapper contract — mirrors the Rust device test."""

from oads.text import AnliR1


def test_anli_r1_decodes_split(engine):
	dataset = AnliR1.load(engine=engine, progress=False)
	assert len(dataset) == 16946
	assert dataset.labels.shape == [16946]

	item = dataset.get_item(0)
	assert item.shape[0] == 1
	x, y = dataset.get_sample(16946 - 1)
	assert x.shape[0] == 1
	assert y.shape == [1]
