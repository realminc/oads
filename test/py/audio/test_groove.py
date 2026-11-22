"""Device-side Groove window dataset contract."""

import oads


def test_groove_test_split_windows(engine):
	dataset = oads.Groove.load(engine, split="test")
	assert len(dataset) > 0
	assert dataset.seq_len == oads.audio.groove.SEQ_LEN
	assert list(dataset.notes.shape)[1] == 3
	x, y = dataset.get_sample(0)
	assert list(x.shape) == [dataset.seq_len, 3]
	assert list(y.shape) == [dataset.seq_len, 3]
	# y[k] == x[k+1] pitch (next-event target).
	xv, yv = x.read_f32(), y.read_f32()
	for i in range(dataset.seq_len - 1):
		assert xv[3 * (i + 1)] == yv[3 * i]
