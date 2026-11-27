//! Device-side CREMA-D dataset contract. Requires a Vulkan device;
//! the pack is fetched once into `OA_DATA_DIR` (or `~/.oads`) and reused.

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
#[ignore = "requires a Vulkan device and network or cached crema_d pack"]
fn crema_d_loads_test_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::CremaD::load_on(
		&engine,
		&oads::CremaDConfig {
			split: oads::CremaDSplit::Test,
			..Default::default()
		},
	)?;
	assert!(dataset.len() > 0);
	assert_eq!(dataset.split(), oads::CremaDSplit::Test);
	assert_eq!(dataset.waveforms().shape()[0] as i64, dataset.len());
	assert_eq!(dataset.labels().shape(), &[dataset.len() as usize]);

	// Labels are integral emotion indices in 0..=5 into CLASSES.
	let labels = dataset.labels().read::<f32>()?;
	assert!(
		labels
			.iter()
			.all(|v| *v == v.trunc() && (0.0..=5.0).contains(v))
	);

	// lengths[i] never exceeds the padded extent.
	let t = dataset.waveforms().shape()[2];
	assert!(
		dataset
			.lengths()
			.read::<f32>()?
			.iter()
			.all(|v| *v > 0.0 && *v <= t as f32)
	);

	let batch = oa::data::collate(&[dataset.get_sample(0)?, dataset.get_sample(1)?])?;
	assert_eq!(batch.x.shape()[0], 2);
	Ok(())
}
