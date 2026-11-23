//! Device-side STSB GLUE contract. Requires a Vulkan device; the
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
#[ignore = "requires a Vulkan device and network or cached glue_stsb pack"]
fn glue_stsb_decodes_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::GlueStsb::load_on(&engine, &oads::GlueStsbConfig::default())?;
	assert_eq!(dataset.len(), 5749);
	assert_eq!(dataset.labels().shape(), &[5749]);
	assert!(dataset.text2().is_some());

	let item = dataset.get_item(0)?;
	assert_eq!(item.shape()[0], 2);
	let sample = dataset.get_sample(5749 - 1)?;
	assert_eq!(sample.x.shape()[0], 2);
	Ok(())
}
