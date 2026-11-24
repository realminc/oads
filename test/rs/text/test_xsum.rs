//! Device-side xsum contract. Requires a Vulkan device; the pack is
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
#[ignore = "requires a Vulkan device and network or cached xsum pack"]
fn xsum_decodes_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Xsum::load_on(
		&engine,
		&oads::XsumConfig {
			count: Some(4096),
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 4096);
	assert_eq!(dataset.document_lengths().shape(), &[4096]);
	assert_eq!(dataset.summary().shape()[0], 4096);
	let item = dataset.get_item(0)?;
	assert_eq!(item.shape().len(), 1);
	assert!(item.shape()[0] > 0);
	Ok(())
}
