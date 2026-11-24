"""lm1b Python wrapper contract — mirrors the Rust device test."""

from oads.text import Lm1b


def test_lm1b_decodes_split(engine):
	dataset = Lm1b.load(engine=engine, count=100000, progress=False)
	assert len(dataset) == 100000
	assert dataset.sentence_lengths.shape[0] == len(dataset)
	item = dataset.get_item(0)
	assert len(item.shape) == 1
	assert item.shape[0] > 0
