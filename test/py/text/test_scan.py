"""scan Python wrapper contract — mirrors the Rust device test."""

from oads.text import Scan


def test_scan_decodes_split(engine):
	dataset = Scan.load(engine=engine, progress=False)
	assert len(dataset) == 16728
	assert dataset.command.shape[0] == 16728
	assert dataset.labels is None
	item = dataset.get_item(0)
	assert item.shape[0] == 1
