"""stanford_dogs Python wrapper contract — mirrors the Rust device test."""

import oa

from oads.image import StanfordDogs


def test_stanford_dogs_indexes_and_decodes_lazily(engine):
	dataset = StanfordDogs.load(engine=engine, size=(224, 224), progress=False)
	assert len(dataset) == 20_580
	assert len(dataset.classes) == 120
	assert dataset.labels.shape == [20_580]

	item = dataset.get_item(0)
	assert item.shape == [3, 224, 224]
	x, y = dataset.get_sample(20_579)
	assert x.shape == [3, 224, 224]
	assert y.shape == [1]
