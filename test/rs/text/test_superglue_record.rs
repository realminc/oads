//! Device-side superglue_record contract. Requires a Vulkan device; the
//! pack is fetched once into `OA_DATA_DIR` (or `~/.oads`) and reused.

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
#[ignore = "requires a Vulkan device and network or cached superglue_record pack"]
fn superglue_record_decodes_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::SuperglueRecord::load_on(&engine, &oads::SuperglueRecordConfig::default())?;
	assert_eq!(dataset.len(), 100730);
	assert!(dataset.labels().is_none());
	let item = dataset.get_item(0)?;
	assert_eq!(item.shape()[0], 1);
	Ok(())
}
