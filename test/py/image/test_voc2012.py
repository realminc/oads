"""voc2012 Python wrapper contract — mirrors the Rust device test."""

from oads.image import Voc2012


def test_voc2012_indexes_and_decodes_lazily(engine):
	dataset = Voc2012.load(engine=engine, progress=False)
	assert len(dataset) == 5_717
	assert len(dataset.classes) == 20
	assert dataset.classes[0] == "aeroplane"

	x, y = dataset.get_sample(0)
	assert x.shape[0] == 3
	assert y.shape[1] == 5
	x, y = dataset.get_sample(5_716)
	assert y.shape[1] == 5
