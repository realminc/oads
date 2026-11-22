//! Device-side Titanic dataset contract. Requires a Vulkan device;
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
#[ignore = "requires a Vulkan device and network or cached titanic pack"]
fn titanic_loads_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Titanic::load_on(&engine, &oads::TitanicConfig::default())?;
	assert_eq!(dataset.len(), 891);
	assert_eq!(dataset.features().shape(), &[891, 9]);
	assert_eq!(dataset.labels().shape(), &[891]);

	// Labels are binary survival flags; normalized features stay in [0, 1].
	let labels = dataset.labels().read::<f32>()?;
	assert!(labels.iter().all(|v| *v == 0.0 || *v == 1.0));
	let features = dataset.features().read::<f32>()?;
	assert!(features.iter().all(|v| (0.0..=1.0).contains(v)));

	// Row 0 is Braund: pclass 3, male, 22, sibsp 1, parch 0, fare 7.25, S.
	// Min-max normalized pclass 3 -> 1.0 (classes run 1..=3).
	let first = dataset.get_item(0)?.read::<f32>()?;
	assert_eq!(first.len(), 9);
	assert_eq!(first[0], 1.0);
	assert_eq!(first[1], 0.0);

	let batch = oa::data::collate(&[dataset.get_sample(0)?, dataset.get_sample(1)?])?;
	assert_eq!(batch.x.shape(), &[2, 9]);
	Ok(())
}
