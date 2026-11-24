"""winogrande Python wrapper contract — mirrors the Rust device test."""

from oads.text import Winogrande


def test_winogrande_decodes_split(engine):
	dataset = Winogrande.load(engine=engine, split="validation", progress=False)
	assert len(dataset) == 1267
	assert dataset.labels.shape == [1267]

	item = dataset.get_item(0)
	assert item.shape[0] == 1
	x, y = dataset.get_sample(1267 - 1)
	assert x.shape[0] == 1
	assert y.shape == [1]
