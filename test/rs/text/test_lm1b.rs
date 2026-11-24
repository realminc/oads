//! Device-side lm1b contract. Requires a Vulkan device; the pack is
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
#[ignore = "requires a Vulkan device and network or cached lm1b pack"]
fn lm1b_decodes_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Lm1b::load_on(
		&engine,
		&oads::Lm1bConfig {
			count: Some(100_000),
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 100_000);
	assert_eq!(dataset.sentence_lengths().shape()[0] as i64, dataset.len());
	let item = dataset.get_item(0)?;
	assert_eq!(item.shape().len(), 1);
	assert!(item.shape()[0] > 0);
	Ok(())
}
