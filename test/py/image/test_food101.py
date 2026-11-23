"""food101 Python wrapper contract — mirrors the Rust device test."""

import oa

from oads.image import Food101


def test_food101_indexes_and_decodes_lazily(engine):
	dataset = Food101.load(
		split="train", engine=engine, size=(224, 224), progress=False
	)
	assert len(dataset) == 75_750
	assert len(dataset.classes) == 101
	assert dataset.labels.shape == [75_750]

	item = dataset.get_item(0)
	assert item.shape == [3, 224, 224]
	x, y = dataset.get_sample(75_749)
	assert x.shape == [3, 224, 224]
	assert y.shape == [1]

	test = Food101.load(
		split="test", engine=engine, size=(224, 224), progress=False
	)
	assert len(test) == 25_250
	assert test.classes == dataset.classes
	assert test.get_item(0).shape == [3, 224, 224]
