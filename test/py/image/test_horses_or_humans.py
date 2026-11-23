"""horses_or_humans Python wrapper contract — mirrors the Rust device test."""

import oa

from oads.image import HorsesOrHumans


def test_horses_or_humans_indexes_and_decodes_lazily(engine):
	dataset = HorsesOrHumans.load(
		split="train", engine=engine, size=(224, 224), progress=False
	)
	assert len(dataset) == 1_027
	assert dataset.classes == ["horses", "humans"]
	assert dataset.labels.shape == [1_027]
	assert dataset.label(0) == 0

	item = dataset.get_item(0)
	assert item.shape == [3, 224, 224]
	x, y = dataset.get_sample(1_026)
	assert x.shape == [3, 224, 224]
	assert y.shape == [1]

	validation = HorsesOrHumans.load(
		split="validation", engine=engine, size=(224, 224), progress=False
	)
	assert len(validation) == 256
	assert validation.classes == ["horses", "humans"]
	assert validation.get_item(0).shape == [3, 224, 224]
