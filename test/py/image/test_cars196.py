"""cars196 Python wrapper contract — mirrors the Rust device test."""

import oa

from oads.image import Cars196


def test_cars196_indexes_and_decodes_lazily(engine):
	dataset = Cars196.load(engine=engine, size=(224, 224), progress=False)
	assert len(dataset) == 8_144
	assert len(dataset.classes) == 196
	assert dataset.labels.shape == [8_144]

	item = dataset.get_item(0)
	assert item.shape == [3, 224, 224]
	x, y = dataset.get_sample(8_143)
	assert x.shape == [3, 224, 224]
	assert y.shape == [1]
