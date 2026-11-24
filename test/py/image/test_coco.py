"""coco Python wrapper contract — mirrors the Rust device test."""

from oads.image import Coco


def test_coco_indexes_and_decodes_lazily(engine):
	dataset = Coco.load(engine=engine, split="validation", progress=False)
	assert len(dataset) == 5_000
	assert len(dataset.classes) == 80
	assert dataset.classes[0] == "person"

	x, y = dataset.get_sample(0)
	assert x.shape[0] == 3
	assert y.shape[1] == 5
