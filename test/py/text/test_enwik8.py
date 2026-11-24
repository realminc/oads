"""enwik8 Python wrapper contract — mirrors the Rust device test."""

from oads.text import Enwik8


def test_enwik8_slices_test_range(engine):
	dataset = Enwik8.load(engine=engine, split="test", progress=False)
	assert len(dataset) == 5_000_000 - 128
	assert dataset.tokens.shape == [5_000_000]

	item = dataset.get_item(0)
	assert item.shape == [128]
	x, y = dataset.get_sample(0)
	assert x.shape == [128]
	assert y.shape == [128]
