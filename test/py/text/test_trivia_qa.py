"""trivia_qa Python wrapper contract — mirrors the Rust device test."""

from oads.text import TriviaQa


def test_trivia_qa_decodes_split(engine):
	dataset = TriviaQa.load(engine=engine, progress=False)
	assert len(dataset) == 61888
	assert dataset.question.shape[0] == 61888
	assert dataset.answer is not None
	assert dataset.labels is None
	item = dataset.get_item(0)
	assert item.shape[0] == 1
