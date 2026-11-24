"""wiki40b Python wrapper contract — mirrors the Rust device test."""

from oads.text import Wiki40B


def test_wiki40b_decodes_split(engine):
	dataset = Wiki40B.load(engine=engine, split="validation", count=1000, progress=False)
	assert len(dataset) == 1000
	assert dataset.article_lengths.shape[0] == len(dataset)
	assert dataset.wikidata_id_lengths.shape[0] == len(dataset)
	item = dataset.get_item(0)
	assert len(item.shape) == 1
	assert item.shape[0] > 0
