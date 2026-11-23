"""oxford_iiit_pet Python wrapper contract — mirrors the Rust device test."""

import oa

from oads.image import OxfordIiitPet


def test_oxford_iiit_pet_indexes_and_decodes_lazily(engine):
	dataset = OxfordIiitPet.load(engine=engine, size=(224, 224), progress=False)
	assert len(dataset) == 7_390
	assert len(dataset.classes) == 37
	assert dataset.labels.shape == [7_390]

	item = dataset.get_item(0)
	assert item.shape == [3, 224, 224]
	x, y = dataset.get_sample(7_389)
	assert x.shape == [3, 224, 224]
	assert y.shape == [1]
