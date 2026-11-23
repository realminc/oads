"""gtzan Python wrapper contract — mirrors the Rust device test."""

from oads.audio import Gtzan


def test_gtzan_decodes_clips_lazily(engine):
	dataset = Gtzan.load(engine=engine, progress=False)
	assert len(dataset) == 999
	assert len(dataset.genres) == 10
	assert dataset.genre_ids.shape == [999]
	assert dataset.lengths.shape == [999]

	waveform = dataset.get_item(0)
	assert waveform.shape[0] == 1
	x, y = dataset.get_sample(0)
	assert x.shape[0] == 1
	assert y.shape == [1]
