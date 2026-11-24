"""cfq Python wrapper contract — mirrors the Rust device test."""

from oads.text import Cfq


def test_cfq_decodes_split(engine):
	dataset = Cfq.load(engine=engine, progress=False)
	assert len(dataset) == 95743
	assert dataset.question.shape[0] == 95743
	assert dataset.labels is None
	item = dataset.get_item(0)
	assert item.shape[0] == 1
