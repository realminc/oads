"""uc_merced Python wrapper contract — mirrors the Rust device test."""

import oa

from oads.image import UcMerced


def test_uc_merced_indexes_and_decodes_lazily(engine):
	dataset = UcMerced.load(engine=engine, size=(224, 224), progress=False)
	assert len(dataset) == 2_100
	assert len(dataset.classes) == 21
	assert dataset.classes[0] == "agricultural"
	assert dataset.labels.shape == [2_100]

	item = dataset.get_item(0)
	assert item.shape == [3, 224, 224]
	x, y = dataset.get_sample(2_099)
	assert x.shape == [3, 224, 224]
	assert y.shape == [1]
