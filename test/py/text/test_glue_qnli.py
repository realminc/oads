"""glue_qnli Python wrapper contract — mirrors the Rust device test."""

from oads.text import GlueQnli


def test_glue_qnli_decodes_split_and_collates(engine):
	dataset = GlueQnli.load(engine=engine, progress=False)
	assert len(dataset) == 104743
	assert dataset.labels.shape == [104743]

	item = dataset.get_item(0)
	assert item.shape[0] == 2
	x, y = dataset.get_sample(104743 - 1)
	assert x.shape[0] == 2
	assert y.shape == [1]
