"""coco_captions Python wrapper contract — mirrors the Rust device test."""

from oads.image import CocoCaptions


def test_coco_captions_indexes_and_decodes_lazily(engine):
	dataset = CocoCaptions.load(engine=engine, split="validation", progress=False)
	assert len(dataset) == 25_014

	x, y = dataset.get_sample(0)
	assert x.shape[0] == 3
	assert len(y.shape) == 1
