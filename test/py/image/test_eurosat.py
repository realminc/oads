"""eurosat Python wrapper contract — mirrors the Rust device test."""

import oa

from oads.image import Eurosat


def test_eurosat_indexes_and_decodes_lazily(engine):
	dataset = Eurosat.load(engine=engine, size=(224, 224), progress=False)
	assert len(dataset) == 27_000
	assert len(dataset.classes) == 10
	assert dataset.classes[0] == "AnnualCrop"
	assert dataset.labels.shape == [27_000]

	item = dataset.get_item(0)
	assert item.shape == [3, 224, 224]
	x, y = dataset.get_sample(26_999)
	assert x.shape == [3, 224, 224]
	assert y.shape == [1]
