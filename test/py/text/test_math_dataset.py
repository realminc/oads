"""math_dataset Python wrapper contract — mirrors the Rust device test."""

from oads.text import MathDataset


def test_math_dataset_decodes_split(engine):
	dataset = MathDataset.load(engine=engine, count=8192, progress=False)
	assert len(dataset) == 8192
	assert dataset.labels.shape[0] == len(dataset)
	assert len(dataset.classes) == 8
	item = dataset.get_item(0)
	assert item.shape[0] == 1
