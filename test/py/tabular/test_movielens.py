"""Device-side MovieLens dataset contracts."""

import oads


def test_100k_test_split_loads(engine):
	dataset = oads.Movielens100k.load(engine, split="test")
	assert len(dataset) == 20_000
	assert list(dataset.features.shape) == [20_000, 3]
	ratings = dataset.ratings.read_f32()
	assert all(v == int(v) and 1 <= v <= 5 for v in ratings)


def test_1m_loads_all_rows(engine):
	dataset = oads.Movielens1m.load(engine)
	assert len(dataset) == 1_000_209
	assert list(dataset.features.shape) == [1_000_209, 3]
