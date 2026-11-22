//! Device-side Fashion-MNIST dataset contract. Requires a Vulkan device;
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
#[ignore = "requires a Vulkan device and network or cached fashion_mnist pack"]
fn fashion_mnist_loads_test_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::FashionMnist::load_on(
		&engine,
		&oads::FashionMnistConfig {
			split: oads::IdxSplit::Test,
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 10_000);
	assert_eq!(dataset.split(), oads::IdxSplit::Test);
	assert_eq!(dataset.images().shape(), &[10_000, 1, 28, 28]);
	assert_eq!(dataset.labels().shape(), &[10_000]);

	// Labels are integral class indices in 0..=9.
	let labels = dataset.labels().read::<f32>()?;
	assert!(
		labels
			.iter()
			.all(|label| label.fract() == 0.0 && (0.0..10.0).contains(label))
	);

	// Per-sample CHW slices collate into [B, 1, 28, 28] NCHW batches.
	let first = dataset.get_sample(0)?;
	let second = dataset.get_sample(1)?;
	assert_eq!(first.x.shape(), &[1, 28, 28]);
	let batch = oa::data::collate(&[first, second])?;
	assert_eq!(batch.x.shape(), &[2, 1, 28, 28]);
	assert_eq!(batch.y.as_ref().expect("labels").shape(), &[2, 1]);

	// Normalized pixels land in [0, 1].
	let pixels = dataset.images().read::<f32>()?;
	assert!(pixels.iter().all(|&v| (0.0..=1.0).contains(&v)));
	Ok(())
}
