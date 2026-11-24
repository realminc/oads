"""go_emotions Python wrapper contract — mirrors the Rust device test."""

from oads.text import GoEmotions


def test_go_emotions_loads_split(engine):
	dataset = GoEmotions.load(engine=engine, split="validation", progress=False)
	assert len(dataset) == 5_426
	assert dataset.labels.shape == [5_426, 28]

	x, y = dataset.get_sample(0)
	assert y.shape[1] == 28
