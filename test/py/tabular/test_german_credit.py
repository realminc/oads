"""german_credit Python wrapper contract — mirrors the Rust device test."""

from oads.tabular import GermanCredit


def test_german_credit_decodes_split(engine):
	dataset = GermanCredit.load(engine=engine, progress=False)
	assert len(dataset) == 1000
	assert list(dataset.features.shape) == [1000, 24]
	labels = dataset.labels.read_f32()
	assert all(v in (0.0, 1.0) for v in labels)
	x, y = dataset.get_sample(0)
	assert list(x.shape) == [24]
