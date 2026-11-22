//! Device-side STL-10 dataset contract. Requires a Vulkan device;
//! the pack is fetched once into `OA_DATA_DIR` (or `~/.oads`) and reused.

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
#[ignore = "requires a Vulkan device and network or cached stl10 pack"]
fn stl10_loads_test_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Stl10::load_on(
		&engine,
		&oads::Stl10Config {
			split: oads::Stl10Split::Test,
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 8_000);
	assert_eq!(dataset.split(), oads::Stl10Split::Test);
	assert_eq!(dataset.images().shape(), &[8_000, 3, 96, 96]);
	assert_eq!(dataset.labels().shape(), &[8_000]);

	// Labels are integral class indices in 0..=9 into CLASSES.
	let labels = dataset.labels().read::<f32>()?;
	assert!(
		labels
			.iter()
			.all(|v| *v == v.trunc() && (0.0..=9.0).contains(v))
	);

	let batch = oa::data::collate(&[dataset.get_sample(0)?, dataset.get_sample(1)?])?;
	assert_eq!(batch.x.shape(), &[2, 3, 96, 96]);
	Ok(())
}
