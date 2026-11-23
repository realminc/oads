"""glue_mnli Python wrapper contract — mirrors the Rust device test."""

from oads.text import GlueMnli


def test_glue_mnli_decodes_split_and_collates(engine):
	dataset = GlueMnli.load(engine=engine, progress=False)
	assert len(dataset) == 392702
	assert dataset.labels.shape == [392702]

	item = dataset.get_item(0)
	assert item.shape[0] == 2
	x, y = dataset.get_sample(392702 - 1)
	assert x.shape[0] == 2
	assert y.shape == [1]
