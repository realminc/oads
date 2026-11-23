//! Device-side beans dataset contract. Requires a Vulkan device;
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
#[ignore = "requires a Vulkan device and network or cached beans pack"]
fn beans_indexes_and_decodes_lazily() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Beans::load_on(
		&engine,
		&oads::BeansConfig {
			split: oads::BeansSplit::Train,
			size: Some((224, 224)),
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 1_034);
	assert_eq!(
		dataset.classes(),
		&["angular_leaf_spot", "bean_rust", "healthy"]
	);
	assert_eq!(dataset.labels().shape(), &[1_034]);

	let labels = dataset.labels().read::<f32>()?;
	assert!(labels.iter().all(|label| (0.0..3.0).contains(label)));

	let item = dataset.get_item(0)?;
	assert_eq!(item.shape(), &[3, 224, 224]);
	let sample = dataset.get_sample(1_033)?;
	let batch = oa::data::collate(&[sample])?;
	assert_eq!(batch.x.shape(), &[1, 3, 224, 224]);
	assert_eq!(batch.y.as_ref().expect("labels").shape(), &[1, 1]);

	let validation = oads::Beans::load_on(
		&engine,
		&oads::BeansConfig {
			split: oads::BeansSplit::Validation,
			size: Some((224, 224)),
			..Default::default()
		},
	)?;
	assert_eq!(validation.len(), 133);
	assert_eq!(validation.classes(), dataset.classes());

	let test = oads::Beans::load_on(
		&engine,
		&oads::BeansConfig {
			split: oads::BeansSplit::Test,
			size: Some((224, 224)),
			..Default::default()
		},
	)?;
	assert_eq!(test.len(), 128);
	assert_eq!(test.get_item(0)?.shape(), &[3, 224, 224]);
	Ok(())
}
