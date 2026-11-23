"""svhn Python wrapper contract — mirrors the Rust device test."""

from oads.image import Svhn


def test_svhn_decodes_split_and_collates(engine):
	dataset = Svhn.load(engine=engine, progress=False)
	assert len(dataset) == 73_257
	assert dataset.images.shape[1:] == [3, 32, 32]
	assert dataset.labels.shape == [73_257]

	item = dataset.get_item(0)
	assert item.shape == [3, 32, 32]
	x, y = dataset.get_sample(73_256)
	assert x.shape == [3, 32, 32]
