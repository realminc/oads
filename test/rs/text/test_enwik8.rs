//! Device-side enwik8 contract. Requires a Vulkan device; the
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
#[ignore = "requires a Vulkan device and network or cached enwik8 pack"]
fn enwik8_slices_test_range() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Enwik8::load_on(
		&engine,
		&oads::Enwik8Config {
			split: oads::Enwik8Split::Test,
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 5_000_000 - 128);
	assert_eq!(dataset.tokens().shape(), &[5_000_000]);
	let sample = dataset.get_sample(0)?;
	assert_eq!(sample.x.shape(), &[128]);
	assert_eq!(sample.y.expect("label window").shape(), &[128]);
	let tokens = dataset.tokens().read::<f32>()?;
	assert!(
		tokens
			.iter()
			.all(|&v| v.fract() == 0.0 && (0.0..=255.0).contains(&v))
	);
	Ok(())
}
