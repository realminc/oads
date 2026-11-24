"""xsum Python wrapper contract — mirrors the Rust device test."""

from oads.text import Xsum


def test_xsum_decodes_split(engine):
	dataset = Xsum.load(engine=engine, count=4096, progress=False)
	assert len(dataset) == 4096
	assert dataset.document_lengths.shape == [4096]
	assert dataset.summary.shape[0] == 4096
	item = dataset.get_item(0)
	assert len(item.shape) == 1
	assert item.shape[0] > 0
