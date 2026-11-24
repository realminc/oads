"""kitti Python wrapper contract — mirrors the Rust device test."""

from oads.image import Kitti


def test_kitti_indexes_and_decodes_lazily(engine):
	dataset = Kitti.load(engine=engine, progress=False)
	assert len(dataset) == 7_481
	assert "Car" in dataset.classes
	assert "DontCare" not in dataset.classes

	x, y = dataset.get_sample(0)
	assert x.shape[0] == 3
	assert y.shape[1] == 5


def test_kitti_test_split_is_unlabeled(engine):
	dataset = Kitti.load(engine=engine, split="test", progress=False)
	assert len(dataset) == 7_518
	x, y = dataset.get_sample(0)
	assert y.shape == [0, 5]
