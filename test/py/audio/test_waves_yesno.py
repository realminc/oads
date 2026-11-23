"""waves_yesno Python wrapper contract — mirrors the Rust device test."""

from oads.audio import WavesYesno


def test_waves_yesno_decodes_and_collates(engine):
	dataset = WavesYesno.load(engine=engine, progress=False)
	assert len(dataset) == 60
	assert dataset.labels.shape == [60, 8]
	assert dataset.lengths.shape == [60]
	assert dataset.waveforms.shape[0] == 60
	assert dataset.waveforms.shape[1] == 1

	item = dataset.get_item(0)
	assert item.shape[0] == 1
	x, y = dataset.get_sample(59)
	assert y.shape[1] == 8
