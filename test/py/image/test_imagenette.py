"""imagenette Python wrapper contract — mirrors the Rust device test."""

import oa

from oads.image import Imagenette


def test_imagenette_indexes_and_decodes_lazily(engine):
	dataset = Imagenette.load(
		split="train", engine=engine, size=(224, 224), progress=False
	)
	assert len(dataset) == 9_469
	assert len(dataset.classes) == 10
	assert dataset.labels.shape == [9_469]
	assert dataset.label(0) == 0

	item = dataset.get_item(0)
	assert item.shape == [3, 224, 224]
	x, y = dataset.get_sample(9_468)
	assert x.shape == [3, 224, 224]
	assert y.shape == [1]

	validation = Imagenette.load(
		split="validation", engine=engine, size=(224, 224), progress=False
	)
	assert len(validation) == 3_925
	assert validation.classes == dataset.classes
	assert validation.get_item(0).shape == [3, 224, 224]
