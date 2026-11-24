"""voc2007 Python wrapper contract — mirrors the Rust device test."""

from oads.image import Voc2007


def test_voc2007_indexes_and_decodes_lazily(engine):
	dataset = Voc2007.load(engine=engine, progress=False)
	assert len(dataset) == 2_501
	assert len(dataset.classes) == 20
	assert dataset.classes[0] == "aeroplane"

	x, y = dataset.get_sample(0)
	assert x.shape[0] == 3
	assert y.shape[1] == 5
	assert dataset.box_count(2_500) >= 0
	x, y = dataset.get_sample(2_500)
	assert y.shape[1] == 5
