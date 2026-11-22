"""Device-side VCTK dataset contract."""

import oads


def test_vctk_loads_speakers_and_lazy_waveforms(engine):
	dataset = oads.Vctk.load(engine)
	assert len(dataset) > 40_000
	assert len(dataset.speakers) > 0
	assert dataset.speaker_ids.shape[0] == len(dataset)
	assert dataset.lengths.shape[0] == len(dataset)

	ids = dataset.speaker_ids.read_f32()
	assert all(v == int(v) and 0 <= v < len(dataset.speakers) for v in ids)

	lengths = dataset.lengths.read_f32()
	wave = dataset.waveform(0)
	assert wave.shape[0] == 1
	assert wave.shape[1] == int(lengths[0])
