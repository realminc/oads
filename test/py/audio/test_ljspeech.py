"""Device-side LJSpeech dataset contract."""

import oads


def test_transcripts_and_lazy_waveforms(engine):
	dataset = oads.LjSpeech.load(engine)
	assert len(dataset) == 13_100
	assert dataset.transcripts.shape[0] == 13_100
	lengths = dataset.audio_lengths.read_f32()
	wave = dataset.waveform(0)
	assert list(wave.shape) == [1, int(lengths[0])]
	x, y = dataset.get_sample(1)
	assert list(y.shape) == [1, int(lengths[1])]
