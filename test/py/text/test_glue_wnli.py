"""glue_wnli Python wrapper contract — mirrors the Rust device test."""

from oads.text import GlueWnli


def test_glue_wnli_decodes_split_and_collates(engine):
	dataset = GlueWnli.load(engine=engine, progress=False)
	assert len(dataset) == 635
	assert dataset.labels.shape == [635]

	item = dataset.get_item(0)
	assert item.shape[0] == 2
	x, y = dataset.get_sample(635 - 1)
	assert x.shape[0] == 2
	assert y.shape == [1]
