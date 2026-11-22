"""Device-side Moving MNIST dataset contract."""

import oads

FRAME = 64 * 64


def test_sequences_shift_targets_by_one_frame(engine):
	dataset = oads.MovingMnist.load(engine, count=8)
	assert len(dataset) == 8
	assert dataset.seq_len == 20

	# Frames are stored flat [N*T, 1, 64, 64]; a sample is a T-frame slice.
	assert list(dataset.frames.shape) == [8 * 20, 1, 64, 64]
	assert list(dataset.get_item(0).shape) == [20, 1, 64, 64]

	x, y = dataset.get_sample(0)
	assert list(x.shape) == [19, 1, 64, 64]
	assert list(y.shape) == [19, 1, 64, 64]
	xt, yt = x.read_f32(), y.read_f32()
	assert all(yt[k] == xt[k + FRAME] for k in range(FRAME))

	frames = dataset.frames.read_f32()
	assert all(0.0 <= v <= 1.0 for v in frames[: 4 * FRAME])
