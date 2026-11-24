"""piqa Python wrapper contract — mirrors the Rust device test."""

from oads.text import Piqa


def test_piqa_decodes_split(engine):
	dataset = Piqa.load(engine=engine, split="validation", progress=False)
	assert len(dataset) == 1838
	assert dataset.labels.shape == [1838]

	item = dataset.get_item(0)
	assert item.shape[0] == 1
	x, y = dataset.get_sample(1838 - 1)
	assert x.shape[0] == 1
	assert y.shape == [1]
