"""nsynth Python wrapper contract — mirrors the Rust device test."""

import oads


def test_nsynth_loads_families_and_lazy_waveforms(engine):
	dataset = oads.Nsynth.load(engine=engine, progress=False)
	assert len(dataset) == 4_096
	assert len(dataset.families) == 11
	assert dataset.labels.shape[0] == len(dataset)
	assert dataset.pitches.shape[0] == len(dataset)

	labels = dataset.labels.read_f32()
	assert all(v == int(v) and 0 <= v < 11 for v in labels)

	lengths = dataset.lengths.read_f32()
	assert all(v == 64_000 for v in lengths)
	wave = dataset.waveform(0)
	assert wave.shape == [1, 64_000]

	x, y = dataset.get_sample(4_095)
	assert x.shape[0] == 1
	assert y.shape == [1]
