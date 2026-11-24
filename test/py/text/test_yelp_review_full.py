"""yelp_review_full Python wrapper contract — mirrors the Rust device test."""

from oads.text import YelpReviewFull


def test_yelp_review_full_loads_test_split(engine):
	dataset = YelpReviewFull.load(engine=engine, split="test", progress=False)
	assert len(dataset) == 50_000
	assert dataset.labels.shape == [50_000]

	x, y = dataset.get_sample(0)
	assert y.shape == [1]
