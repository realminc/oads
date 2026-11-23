"""cub200 Python wrapper contract — mirrors the Rust device test."""

import oa

from oads.image import Cub200


def test_cub200_indexes_and_decodes_lazily(engine):
	dataset = Cub200.load(engine=engine, size=(224, 224), progress=False)
	assert len(dataset) == 11_788
	assert len(dataset.classes) == 200
	assert dataset.classes[0] == "001.Black_footed_Albatross"
	assert dataset.labels.shape == [11_788]

	item = dataset.get_item(0)
	assert item.shape == [3, 224, 224]
	x, y = dataset.get_sample(11_787)
	assert x.shape == [3, 224, 224]
	assert y.shape == [1]
