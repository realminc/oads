//! Device-side SVHN contract. Requires a Vulkan device; the pack is
//! fetched once into `OA_DATA_DIR` (or `~/.oads`) and reused.

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
#[ignore = "requires a Vulkan device and network or cached svhn pack"]
fn svhn_decodes_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Svhn::load_on(&engine, &oads::SvhnConfig::default())?;
	assert_eq!(dataset.len(), 73_257);
	assert_eq!(dataset.images().shape()[1..], [3, 32, 32]);
	assert_eq!(dataset.labels().shape(), &[73_257]);

	let labels = dataset.labels().read::<f32>()?;
	assert!(
		labels
			.iter()
			.all(|&v| v == v.trunc() && (0.0..10.0).contains(&v))
	);
	let item = dataset.get_item(0)?;
	assert_eq!(item.shape(), &[3, 32, 32]);
	let sample = dataset.get_sample(73_256)?;
	assert_eq!(sample.y.expect("sample carries a label").shape(), &[1]);
	Ok(())
}
