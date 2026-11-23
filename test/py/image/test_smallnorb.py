"""smallnorb Python wrapper contract — mirrors the Rust device test."""

from oads.image import Smallnorb


def test_smallnorb_decodes_split_and_collates(engine):
	dataset = Smallnorb.load(engine=engine, progress=False)
	assert len(dataset) == 24_300
	assert dataset.images.shape[1:] == [2, 96, 96]
	assert dataset.labels.shape == [24_300]
	assert dataset.info.shape == [24_300, 4]
	assert len(dataset.classes) == 5

	item = dataset.get_item(0)
	assert item.shape == [2, 96, 96]
