//! Device-side Places365-small dataset contract. Requires a Vulkan
//! device; the ~25 GB pack is fetched once into `OA_DATA_DIR` (or
//! `~/.oads`) and reused.

use oa::data::Dataset;

fn engine() -> oa::Result<oa::Engine> {
	let selection = match std::env::var("OA_TEST_DEVICE_INDEX") {
		Ok(index) => oa::DeviceSelection::Index(
			index
				.parse()
				.expect("OA_TEST_DEVICE_INDEX must be a nonnegative integer"),
		),
		Err(std::env::VarError::NotPresent) => oa::DeviceSelection::Automatic,
		Err(_) => panic!("OA_TEST_DEVICE_INDEX must be Unicode"),
	};
	let engine = oa::Engine::builder().devices(selection).build()?;
	oa::Engine::set_default(engine.clone());
	Ok(engine)
}

#[test]
#[ignore = "requires a Vulkan device and the cached places365_small pack"]
fn places365_small_indexes_and_decodes_lazily() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Places365Small::load_on(
		&engine,
		&oads::Places365SmallConfig {
			size: Some((224, 224)),
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 1_803_460);
	assert_eq!(dataset.classes().len(), 365);
	assert_eq!(dataset.labels().shape(), &[1_803_460]);

	let labels = dataset.labels().read::<f32>()?;
	assert!(labels.iter().all(|label| (0.0..365.0).contains(label)));

	let item = dataset.get_item(0)?;
	assert_eq!(item.shape(), &[3, 224, 224]);
	let sample = dataset.get_sample(1_803_459)?;
	let batch = oa::data::collate(&[sample])?;
	assert_eq!(batch.x.shape(), &[1, 3, 224, 224]);
	assert_eq!(batch.y.as_ref().expect("labels").shape(), &[1, 1]);

	let validation = oads::Places365Small::load_on(
		&engine,
		&oads::Places365SmallConfig {
			split: oads::Places365SmallSplit::Validation,
			size: Some((224, 224)),
			..Default::default()
		},
	)?;
	assert_eq!(validation.len(), 36_500);
	assert_eq!(validation.get_item(0)?.shape(), &[3, 224, 224]);
	Ok(())
}
