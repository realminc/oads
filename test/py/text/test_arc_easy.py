"""arc_easy Python wrapper contract — mirrors the Rust device test."""

from oads.text import ArcEasy


def test_arc_easy_decodes_split(engine):
	dataset = ArcEasy.load(engine=engine, progress=False)
	assert len(dataset) == 2251
	assert dataset.question.shape[0] == 2251
	assert dataset.labels.shape == [2251]
	assert len(dataset.classes) == 10
	item = dataset.get_item(0)
	assert item.shape[0] == 1
