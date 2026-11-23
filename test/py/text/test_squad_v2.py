"""squad_v2 Python wrapper contract — mirrors the Rust device test."""

from oads.text import SquadV2


def test_squad_v2_decodes_split_and_collates(engine):
	dataset = SquadV2.load(engine=engine, progress=False)
	assert len(dataset) == 130_319
	assert dataset.question_offsets.shape == [130_319]
	assert dataset.question_lengths.shape == [130_319]
	assert dataset.context_index.shape == [130_319]
	assert len(dataset.context_offsets.shape) == 1
	assert dataset.context_offsets.shape[0] < 130_319
	assert dataset.answer_start.shape == [130_319]
	assert dataset.is_impossible.shape == [130_319]
	assert dataset.answer_offsets.shape == [130_319]

	item = dataset.get_item(0)
	assert len(item.shape) == 1
