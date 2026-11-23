"""libritts Python wrapper contract — mirrors the Rust device test."""

import oads


def test_libritts_loads_speakers_and_lazy_waveforms(engine):
	dataset = oads.Libritts.load(engine=engine, progress=False)
	assert len(dataset) == 5_736
	assert len(dataset.speakers) == 40
	assert dataset.speaker_ids.shape[0] == len(dataset)
	assert dataset.lengths.shape[0] == len(dataset)

	ids = dataset.speaker_ids.read_f32()
	assert all(v == int(v) and 0 <= v < len(dataset.speakers) for v in ids)

	lengths = dataset.lengths.read_f32()
	wave = dataset.waveform(0)
	assert wave.shape[0] == 1
	assert wave.shape[1] == int(lengths[0])

	x, y = dataset.get_sample(5_735)
	assert x.shape[0] == 1
	assert y.shape == [1]
