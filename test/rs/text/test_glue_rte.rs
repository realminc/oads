//! Device-side RTE GLUE contract. Requires a Vulkan device; the
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
#[ignore = "requires a Vulkan device and network or cached glue_rte pack"]
fn glue_rte_decodes_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::GlueRte::load_on(&engine, &oads::GlueRteConfig::default())?;
	assert_eq!(dataset.len(), 2490);
	assert_eq!(dataset.labels().shape(), &[2490]);
	assert_eq!(dataset.classes().len(), 2);
	assert!(dataset.text2().is_some());

	let item = dataset.get_item(0)?;
	assert_eq!(item.shape()[0], 2);
	let sample = dataset.get_sample(2490 - 1)?;
	assert_eq!(sample.x.shape()[0], 2);
	Ok(())
}
