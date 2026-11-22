"""Device-side Tiny Shakespeare dataset contract."""

import oads

SEQ_LEN = 64


def test_samples_shift_targets_by_one_byte(engine):
	dataset = oads.TinyShakespeare.load(engine, seq_len=SEQ_LEN)
	assert len(dataset) == dataset.tokens.shape[0] - SEQ_LEN

	x, y = dataset.get_sample(0)
	assert list(x.shape) == [SEQ_LEN]
	assert list(y.shape) == [SEQ_LEN]
	xt, yt = x.read_f32(), y.read_f32()
	assert all(yt[k] == xt[k + 1] for k in range(SEQ_LEN - 1))

	item = dataset.get_item(1)
	assert list(item.shape) == [SEQ_LEN]
	assert item.read_f32()[0] == xt[1]
