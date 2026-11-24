"""squad_v1 Python wrapper contract — mirrors the Rust device test."""

from oads.text import SquadV1


def test_squad_v1_decodes_split(engine):
	dataset = SquadV1.load(engine=engine, split="validation", progress=False)
	assert len(dataset) == 10_570
	assert dataset.question_offsets.shape == [10_570]
	assert dataset.answer_start.shape == [10_570]

	item = dataset.get_item(0)
	assert len(item.shape) == 1
	x, y = dataset.get_sample(0)
	assert y.shape == [1]
