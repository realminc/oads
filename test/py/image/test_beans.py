"""beans Python wrapper contract — mirrors the Rust device test."""

import oa

from oads.image import Beans


def test_beans_indexes_and_decodes_lazily(engine):
	dataset = Beans.load(
		split="train", engine=engine, size=(224, 224), progress=False
	)
	assert len(dataset) == 1_034
	assert dataset.classes == ["angular_leaf_spot", "bean_rust", "healthy"]
	assert dataset.labels.shape == [1_034]

	item = dataset.get_item(0)
	assert item.shape == [3, 224, 224]
	x, y = dataset.get_sample(1_033)
	assert x.shape == [3, 224, 224]
	assert y.shape == [1]

	validation = Beans.load(
		split="validation", engine=engine, size=(224, 224), progress=False
	)
	assert len(validation) == 133
	assert validation.classes == dataset.classes

	test = Beans.load(
		split="test", engine=engine, size=(224, 224), progress=False
	)
	assert len(test) == 128
	assert test.get_item(0).shape == [3, 224, 224]
