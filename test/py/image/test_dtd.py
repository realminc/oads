"""dtd Python wrapper contract — mirrors the Rust device test."""

import oa

from oads.image import Dtd


def test_dtd_indexes_and_decodes_lazily(engine):
	dataset = Dtd.load(
		split="train", engine=engine, size=(224, 224), progress=False
	)
	assert len(dataset) == 1_880
	assert len(dataset.classes) == 47
	assert dataset.classes[0] == "banded"
	assert dataset.labels.shape == [1_880]

	item = dataset.get_item(0)
	assert item.shape == [3, 224, 224]
	x, y = dataset.get_sample(1_879)
	assert x.shape == [3, 224, 224]
	assert y.shape == [1]

	validation = Dtd.load(
		split="validation", engine=engine, size=(224, 224), progress=False
	)
	assert len(validation) == 1_880
	assert validation.classes == dataset.classes

	test = Dtd.load(
		split="test", engine=engine, size=(224, 224), progress=False
	)
	assert len(test) == 1_880
	assert test.get_item(0).shape == [3, 224, 224]
