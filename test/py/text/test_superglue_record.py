"""superglue_record Python wrapper contract — mirrors the Rust device test."""

from oads.text import SuperglueRecord


def test_superglue_record_decodes_split(engine):
	dataset = SuperglueRecord.load(engine=engine, progress=False)
	assert len(dataset) == 100730
	assert dataset.labels is None

	item = dataset.get_item(0)
	assert item.shape[0] == 1
	x, y = dataset.get_sample(100730 - 1)
	assert x.shape[0] == 1
