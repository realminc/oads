"""Device-side MAESTRO contract: [E, 3] (pitch, step, duration) events
windowed as [seq_len, 3] device slices, paired with the one-event-shifted
target window. Requires a Vulkan device; the pack is fetched once into
OA_DATA_DIR (or the platform cache) and reused."""

import oads
from oads.audio import maestro


def test_maestro_loads_windows_and_samples(engine):
	dataset = oads.Maestro.load(engine)
	assert len(dataset) > 0
	assert list(dataset.notes.shape)[1] == 3
	assert dataset.seq_len == maestro.SEQ_LEN
	assert dataset.split == "train"

	x, y = dataset.get_sample(0)
	assert list(x.shape) == [dataset.seq_len, 3]
	assert list(y.shape) == [dataset.seq_len, 3]

	# y is x shifted one event — the next-event prediction contract.
	xv, yv = x.read_f32(), y.read_f32()
	for i in range(dataset.seq_len - 1):
		assert xv[3 * (i + 1)] == yv[3 * i]

	item = dataset.get_item(1)
	assert list(item.shape) == [dataset.seq_len, 3]

	notes = dataset.notes.slice(0, 0, 4_096).read_f32()
	assert all(
		0.0 <= e[0] <= 127.0 and e[1] >= 0.0 and e[2] >= 0.0
		for e in zip(*[iter(notes)] * 3)
	)


def test_maestro_test_split_loads(engine):
	dataset = oads.Maestro.load(engine, split="test")
	assert dataset.split == "test"
	assert len(dataset) > 0
