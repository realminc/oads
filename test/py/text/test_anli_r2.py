"""anli_r2 Python wrapper contract — mirrors the Rust device test."""

from oads.text import AnliR2


def test_anli_r2_decodes_split(engine):
	dataset = AnliR2.load(engine=engine, progress=False)
	assert len(dataset) == 45460
	assert dataset.labels.shape == [45460]

	item = dataset.get_item(0)
	assert item.shape[0] == 1
	x, y = dataset.get_sample(45460 - 1)
	assert x.shape[0] == 1
	assert y.shape == [1]
