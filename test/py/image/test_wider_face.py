"""wider_face Python wrapper contract — mirrors the Rust device test."""

from oads.image import WiderFace


def test_wider_face_indexes_and_decodes_lazily(engine):
	dataset = WiderFace.load(engine=engine, progress=False)
	assert len(dataset) == 12_880
	assert dataset.classes == ["face"]

	x, y = dataset.get_sample(0)
	assert x.shape[0] == 3
	assert y.shape[1] == 5
