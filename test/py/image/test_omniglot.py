"""omniglot Python wrapper contract — mirrors the Rust device test."""

from oads.image import Omniglot


def test_omniglot_decodes_split_and_collates(engine):
	dataset = Omniglot.load(engine=engine, progress=False)
	assert len(dataset) == 19_280
	assert dataset.labels.shape == [19_280]
	assert len(dataset.classes) == 964
	assert "/" in dataset.classes[0]

	item = dataset.get_item(0)
	assert len(item.shape) == 3
