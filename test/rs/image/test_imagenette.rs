//! Device-side Imagenette dataset contract. Requires a Vulkan device;
//! the pack is fetched once into `OA_DATA_DIR` (or `~/.oads`) and
//! reused.

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
#[ignore = "requires a Vulkan device and network or cached imagenette pack"]
fn imagenette_indexes_and_decodes_lazily() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Imagenette::load_on(
		&engine,
		&oads::ImagenetteConfig {
			split: oads::ImagenetteSplit::Train,
			size: Some((224, 224)),
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 9_469);
	assert_eq!(dataset.classes().len(), 10);
	assert_eq!(dataset.labels().shape(), &[9_469]);

	let labels = dataset.labels().read::<f32>()?;
	assert!(labels.iter().all(|label| (0.0..10.0).contains(label)));

	let item = dataset.get_item(0)?;
	assert_eq!(item.shape(), &[3, 224, 224]);
	let sample = dataset.get_sample(9_468)?;
	let batch = oa::data::collate(&[sample])?;
	assert_eq!(batch.x.shape(), &[1, 3, 224, 224]);
	assert_eq!(batch.y.as_ref().expect("labels").shape(), &[1, 1]);

	let validation = oads::Imagenette::load_on(
		&engine,
		&oads::ImagenetteConfig {
			split: oads::ImagenetteSplit::Validation,
			size: Some((224, 224)),
			..Default::default()
		},
	)?;
	assert_eq!(validation.len(), 3_925);
	assert_eq!(validation.classes(), dataset.classes());
	assert_eq!(validation.get_item(0)?.shape(), &[3, 224, 224]);
	Ok(())
}
