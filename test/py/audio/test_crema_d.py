"""Device-side CREMA-D dataset contract."""

import oads


def test_test_split_loads_and_collates(engine):
	dataset = oads.CremaD.load(engine, split="test")
	assert len(dataset) > 0
	assert dataset.waveforms.shape[0] == len(dataset)
	assert dataset.waveforms.shape[1] == 1
	labels = dataset.labels.read_f32()
	assert all(v == int(v) and 0 <= v <= 5 for v in labels)
	lengths = dataset.lengths.read_f32()
	assert all(0 < v <= dataset.waveforms.shape[2] for v in lengths)
