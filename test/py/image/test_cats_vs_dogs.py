"""cats_vs_dogs Python wrapper contract — mirrors the Rust device test."""

import oa

from oads.image import CatsVsDogs


def test_cats_vs_dogs_indexes_and_decodes_lazily(engine):
	dataset = CatsVsDogs.load(engine=engine, size=(224, 224), progress=False)
	assert len(dataset) == 25_000
	assert dataset.classes == ["Cat", "Dog"]
	assert dataset.labels.shape == [25_000]
	assert dataset.label(0) == 0
	assert dataset.label(12_500) == 1

	item = dataset.get_item(0)
	assert item.shape == [3, 224, 224]
	x, y = dataset.get_sample(12_499)
	assert x.shape == [3, 224, 224]
	assert y.shape == [1]
