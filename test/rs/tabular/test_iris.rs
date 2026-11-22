//! Device-side Iris dataset contract. Requires a Vulkan device;
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
#[ignore = "requires a Vulkan device and network or cached iris pack"]
fn iris_loads_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Iris::load_on(&engine, &oads::IrisConfig::default())?;
	assert_eq!(dataset.len(), 150);
	assert_eq!(dataset.features().shape(), &[150, 4]);
	assert_eq!(dataset.labels().shape(), &[150]);

	// Labels are integral class indices in 0..=2 and features are
	// min-max normalized to [0, 1].
	let labels = dataset.labels().read::<f32>()?;
	assert!(
		labels
			.iter()
			.all(|v| *v == v.trunc() && (0.0..=2.0).contains(v))
	);
	let features = dataset.features().read::<f32>()?;
	assert!(features.iter().all(|v| (0.0..=1.0).contains(v)));

	// Fisher's canonical first row is setosa 5.1, 3.5, 1.4, 0.2.
	let setosa = labels.iter().take_while(|v| **v == 0.0).count();
	assert_eq!(setosa, 50);

	let batch = oa::data::collate(&[dataset.get_sample(0)?, dataset.get_sample(1)?])?;
	assert_eq!(batch.x.shape(), &[2, 4]);
	Ok(())
}
