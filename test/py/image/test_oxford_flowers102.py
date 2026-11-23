from oads.image import OxfordFlowers102


def test_oxford_flowers102_indexes_and_decodes_lazily(engine):
	dataset = OxfordFlowers102.load(
		split="train", engine=engine, size=(224, 224), progress=False
	)
	assert len(dataset) == 1_020
	assert len(dataset.classes) == 102
	assert dataset.classes[0] == "pink primrose"
	assert dataset.classes[101] == "blackberry lily"
	assert dataset.labels.shape == [1_020]

	item = dataset.get_item(0)
	assert item.shape == [3, 224, 224]
	x, y = dataset.get_sample(1_019)
	assert x.shape == [3, 224, 224]
	assert y.shape == [1]

	validation = OxfordFlowers102.load(
		split="validation", engine=engine, size=(224, 224), progress=False
	)
	assert len(validation) == 1_020
	assert validation.classes == dataset.classes

	test = OxfordFlowers102.load(
		split="test", engine=engine, size=(224, 224), progress=False
	)
	assert len(test) == 6_149
	assert test.get_item(0).shape == [3, 224, 224]
