"""openbookqa Python wrapper contract — mirrors the Rust device test."""

from oads.text import Openbookqa


def test_openbookqa_decodes_split(engine):
	dataset = Openbookqa.load(engine=engine, split="validation", progress=False)
	assert len(dataset) == 500
	assert dataset.labels.shape == [500]

	item = dataset.get_item(0)
	assert item.shape[0] == 1
	x, y = dataset.get_sample(500 - 1)
	assert x.shape[0] == 1
	assert y.shape == [1]
