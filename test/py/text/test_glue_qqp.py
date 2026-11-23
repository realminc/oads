"""glue_qqp Python wrapper contract — mirrors the Rust device test."""

from oads.text import GlueQqp


def test_glue_qqp_decodes_split_and_collates(engine):
	dataset = GlueQqp.load(engine=engine, progress=False)
	assert len(dataset) == 363846
	assert dataset.labels.shape == [363846]

	item = dataset.get_item(0)
	assert item.shape[0] == 2
	x, y = dataset.get_sample(363846 - 1)
	assert x.shape[0] == 2
	assert y.shape == [1]
