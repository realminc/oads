//! Device-side Tiny Shakespeare dataset contract. Requires a Vulkan device;
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
#[ignore = "requires a Vulkan device and network or cached tiny_shakespeare pack"]
fn tiny_shakespeare_windows_shift_labels_by_one() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::TinyShakespeare::load_on(
		&engine,
		&oads::TinyShakespeareConfig {
			seq_len: 32,
			..Default::default()
		},
	)?;
	assert_eq!(dataset.tokens().shape(), &[dataset.text().len()]);
	assert_eq!(dataset.len(), dataset.text().len() as i64 - 32);
	assert!(dataset.text().starts_with("First Citizen:"));

	let sample = dataset.get_sample(0)?;
	assert_eq!(sample.x.shape(), &[32]);
	assert_eq!(sample.y.as_ref().expect("label").shape(), &[32]);

	// y is x shifted one byte: y[k] == tokens[k + 1] == x[k + 1].
	let x = sample.x.read::<f32>()?;
	let y = sample.y.expect("label").read::<f32>()?;
	assert_eq!(&y[..31], &x[1..]);

	// Engineless load resolves the thread default installed by `engine()`.
	let defaulted = oads::TinyShakespeare::load(&oads::TinyShakespeareConfig {
		seq_len: 32,
		..Default::default()
	})?;
	assert_eq!(defaulted.len(), dataset.len());

	// Token values are exact UTF-8 bytes.
	let tokens = dataset.tokens().read::<f32>()?;
	assert!(
		tokens
			.iter()
			.all(|v| v.fract() == 0.0 && (0.0..=255.0).contains(v))
	);
	Ok(())
}
