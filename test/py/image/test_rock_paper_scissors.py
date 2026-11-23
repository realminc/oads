"""rock_paper_scissors Python wrapper contract — mirrors the Rust device test."""

import oa

from oads.image import RockPaperScissors


def test_rock_paper_scissors_indexes_and_decodes_lazily(engine):
	dataset = RockPaperScissors.load(
		split="train", engine=engine, size=(224, 224), progress=False
	)
	assert len(dataset) == 2_520
	assert dataset.classes == ["paper", "rock", "scissors"]
	assert dataset.labels.shape == [2_520]
	assert dataset.label(0) == 0

	item = dataset.get_item(0)
	assert item.shape == [3, 224, 224]
	x, y = dataset.get_sample(2_519)
	assert x.shape == [3, 224, 224]
	assert y.shape == [1]

	test = RockPaperScissors.load(
		split="test", engine=engine, size=(224, 224), progress=False
	)
	assert len(test) == 372
	assert test.classes == ["paper", "rock", "scissors"]
	assert test.get_item(0).shape == [3, 224, 224]
