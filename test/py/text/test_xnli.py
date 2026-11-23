"""xnli Python wrapper contract — mirrors the Rust device test."""

from oads.text import Xnli


def test_xnli_decodes_validation(engine):
	dataset = Xnli.load(engine=engine, progress=False)
	assert len(dataset) == 37350
	assert dataset.labels.shape == [37350]
	assert dataset.languages.shape == [37350]

	item = dataset.get_item(0)
	assert item.shape[0] == 1
	x, y = dataset.get_sample(37350 - 1)
	assert x.shape[0] == 1
	assert y.shape == [1]
