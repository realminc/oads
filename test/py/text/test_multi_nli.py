"""multi_nli Python wrapper contract — mirrors the Rust device test."""

from oads.text import MultiNli


def test_multi_nli_decodes_split(engine):
	dataset = MultiNli.load(engine=engine, split="validation", progress=False)
	assert len(dataset) == 9815
	assert dataset.labels.shape == [9815]

	item = dataset.get_item(0)
	assert item.shape[0] == 1
	x, y = dataset.get_sample(9815 - 1)
	assert x.shape[0] == 1
	assert y.shape == [1]
