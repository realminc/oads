"""lfw Python wrapper contract — mirrors the Rust device test."""

import oa

from oads.image import Lfw


def test_lfw_indexes_and_decodes_lazily(engine):
	dataset = Lfw.load(engine=engine, size=(224, 224), progress=False)
	assert len(dataset) == 13_233
	assert len(dataset.classes) == 5_749
	assert dataset.labels.shape == [13_233]

	item = dataset.get_item(0)
	assert item.shape == [3, 224, 224]
	x, y = dataset.get_sample(13_232)
	assert x.shape == [3, 224, 224]
	assert y.shape == [1]
