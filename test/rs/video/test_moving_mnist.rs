//! Device-side Moving MNIST dataset contract. Requires a Vulkan device;
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
#[ignore = "requires a Vulkan device and network or cached moving_mnist pack"]
fn moving_mnist_sequences_shift_targets_by_one_frame() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::MovingMnist::load_on(
		&engine,
		&oads::MovingMnistConfig {
			count: Some(64),
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 64);
	assert_eq!(dataset.seq_len(), 20);
	assert_eq!(dataset.frames().shape(), &[64 * 20, 1, 64, 64]);

	let sample = dataset.get_sample(0)?;
	assert_eq!(sample.x.shape(), &[19, 1, 64, 64]);
	assert_eq!(sample.y.as_ref().expect("label").shape(), &[19, 1, 64, 64]);

	// y is x shifted one frame: y[k] == x[k + 1].
	let x = sample.x.read::<f32>()?;
	let y = sample.y.expect("label").read::<f32>()?;
	let frame = 64 * 64;
	assert_eq!(&y[..18 * frame], &x[frame..]);

	// Normalized pixels land in [0, 1] and the sequence is not uniform.
	let pixels = dataset.frames().read::<f32>()?;
	assert!(pixels.iter().all(|&v| (0.0..=1.0).contains(&v)));
	assert!(pixels.iter().any(|&v| v > 0.5));
	Ok(())
}
