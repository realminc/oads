//! Device-side adult contract. Requires a Vulkan device; the
//! pack is fetched once into `OA_DATA_DIR` (or `~/.oads`) and reused.

use oa::data::Dataset;

fn engine() -> oa::Result<oa::Runtime> {
	let selection = match std::env::var("OA_TEST_DEVICE_INDEX") {
		Ok(index) => oa::DeviceSelection::Index(
			index
				.parse()
				.expect("OA_TEST_DEVICE_INDEX must be a nonnegative integer"),
		),
		Err(std::env::VarError::NotPresent) => oa::DeviceSelection::Automatic,
		Err(_) => panic!("OA_TEST_DEVICE_INDEX must be Unicode"),
	};
	let engine = oa::Runtime::builder().devices(selection).build()?;
	oa::Runtime::set_default(engine.clone());
	Ok(engine)
}

#[test]
#[ignore = "requires a Vulkan device and network or cached adult pack"]
fn adult_loads_test_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Adult::load_on(
		&engine,
		&oads::AdultConfig {
			split: oads::AdultSplit::Test,
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 16_281);
	assert_eq!(dataset.features().shape(), &[16_281, 92]);
	assert_eq!(dataset.labels().shape(), &[16_281]);
	let labels = dataset.labels().read::<f32>()?;
	assert!(labels.iter().all(|&v| v == 0.0 || v == 1.0));
	let first = dataset.get_sample(0)?;
	let second = dataset.get_sample(1)?;
	let batch = oa::data::collate(&[first, second])?;
	assert_eq!(batch.x.shape(), &[2, 92]);
	Ok(())
}
