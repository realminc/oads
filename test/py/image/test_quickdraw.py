"""quickdraw Python wrapper contract — mirrors the Rust device test."""

from oads.image import Quickdraw


def test_quickdraw_decodes_count_capped_rows(engine):
	dataset = Quickdraw.load(engine=engine, count=1_000, progress=False)
	assert len(dataset) == 1_000
	assert dataset.x.shape == [1_000, 1, 28, 28]
	assert dataset.labels.shape == [1_000]

	x, y = dataset.get_sample(0)
	assert x.shape == [1, 28, 28]
