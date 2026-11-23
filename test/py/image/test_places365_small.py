from oads.image import Places365Small


def test_places365_small_indexes_and_decodes_lazily(engine):
	dataset = Places365Small.load(
		split="train", engine=engine, size=(224, 224), progress=False
	)
	assert len(dataset) == 1_803_460
	assert len(dataset.classes) == 365
	assert dataset.labels.shape == [1_803_460]

	item = dataset.get_item(0)
	assert item.shape == [3, 224, 224]
	x, y = dataset.get_sample(1_803_459)
	assert x.shape == [3, 224, 224]
	assert y.shape == [1]

	validation = Places365Small.load(
		split="validation", engine=engine, size=(224, 224), progress=False
	)
	assert len(validation) == 36_500
	assert validation.classes == dataset.classes
	assert validation.get_item(0).shape == [3, 224, 224]
