"""imagewang Python wrapper contract — mirrors the Rust device test."""

import oa

from oads.image import Imagewang


def test_imagewang_indexes_and_decodes_lazily(engine):
	dataset = Imagewang.load(
		split="train", engine=engine, size=(224, 224), progress=False
	)
	assert len(dataset) == 14_669
	assert len(dataset.classes) == 20
	assert dataset.labels.shape == [14_669]

	item = dataset.get_item(0)
	assert item.shape == [3, 224, 224]
	x, y = dataset.get_sample(14_668)
	assert x.shape == [3, 224, 224]
	assert y.shape == [1]

	validation = Imagewang.load(
		split="validation", engine=engine, size=(224, 224), progress=False
	)
	assert len(validation) == 3_929
	assert validation.get_item(0).shape == [3, 224, 224]
