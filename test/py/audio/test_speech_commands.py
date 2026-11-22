"""Device-side Speech Commands v0.02 dataset contract."""

import oads


def test_test_split_loads_and_samples(engine):
	dataset = oads.SpeechCommands.load(engine, split="test")
	assert len(dataset) == 11_005
	assert list(dataset.samples.shape) == [11_005, 1, 16_000]
	labels = dataset.labels.read_f32()
	assert all(v == int(v) and 0 <= v < 35 for v in labels)
	x, y = dataset.get_sample(0)
	assert list(x.shape) == [1, 16_000]
	assert len(oads.SpeechCommands.COMMANDS) == 35
