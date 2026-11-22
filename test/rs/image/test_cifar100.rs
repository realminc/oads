//! Device-side CIFAR-100 dataset contract. Requires a Vulkan device;
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
#[ignore = "requires a Vulkan device and network or cached cifar100 pack"]
fn cifar100_loads_test_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Cifar100::load_on(
		&engine,
		&oads::Cifar100Config {
			split: oads::IdxSplit::Test,
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 10_000);
	assert_eq!(dataset.images().shape(), &[10_000, 3, 32, 32]);
	assert_eq!(dataset.labels().shape(), &[10_000]);

	// Fine labels are integral class indices in 0..=99.
	let labels = dataset.labels().read::<f32>()?;
	assert!(
		labels
			.iter()
			.all(|&v| v.fract() == 0.0 && (0.0..=99.0).contains(&v))
	);

	// Per-sample CHW slices collate into [B, 3, 32, 32] NCHW batches.
	let first = dataset.get_sample(0)?;
	let second = dataset.get_sample(1)?;
	assert_eq!(first.x.shape(), &[3, 32, 32]);
	let batch = oa::data::collate(&[first, second])?;
	assert_eq!(batch.x.shape(), &[2, 3, 32, 32]);
	assert_eq!(batch.y.as_ref().expect("labels").shape(), &[2, 1]);

	// Normalized pixels land in [0, 1].
	let pixels = dataset.images().read::<f32>()?;
	assert!(pixels.iter().all(|&v| (0.0..=1.0).contains(&v)));
	Ok(())
}
