"""Device-side mini Speech Commands contract: [N, 1, 16000] f32 waveform
samples and [N] f32 labels over the eight command classes. Requires a
Vulkan device; the pack is fetched once into OA_DATA_DIR (or the platform
cache) and reused."""

import oads


def test_mini_speech_commands_loads_and_samples(engine):
	dataset = oads.MiniSpeechCommands.load(engine)
	assert len(dataset) == 8_000
	assert list(dataset.samples.shape) == [8_000, 1, 16_000]
	assert list(dataset.labels.shape) == [8_000]

	labels = dataset.labels.read_f32()
	assert all(v == int(v) and 0 <= v < 8 for v in labels)
	assert {int(v) for v in labels} == set(range(len(oads.MiniSpeechCommands.COMMANDS)))

	x, y = dataset.get_sample(0)
	assert list(x.shape) == [1, 16_000]
	assert list(y.shape) == [1]

	samples = dataset.samples.slice(0, 0, 256).read_f32()
	assert all(-1.0 <= v <= 1.0 for v in samples)
	assert any(abs(v) > 0.01 for v in samples)
