"""arc_challenge Python wrapper contract — mirrors the Rust device test."""

from oads.text import ArcChallenge


def test_arc_challenge_decodes_split(engine):
	dataset = ArcChallenge.load(engine=engine, progress=False)
	assert len(dataset) == 1119
	assert dataset.question.shape[0] == 1119
	assert dataset.labels.shape == [1119]
	assert len(dataset.classes) == 10
	item = dataset.get_item(0)
	assert item.shape[0] == 1
