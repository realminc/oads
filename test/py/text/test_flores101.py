"""flores101 Python wrapper contract — mirrors the Rust device test."""

from oads.text import Flores101


def test_flores101_decodes_split(engine):
	dataset = Flores101.load(engine=engine, split="validation", progress=False)
	assert len(dataset) == 100697
	assert dataset.english.shape[0] == 100697
	assert dataset.labels.shape == [100697]
	assert len(dataset.classes) == 101
	item = dataset.get_item(0)
	assert item.shape[0] == 1
