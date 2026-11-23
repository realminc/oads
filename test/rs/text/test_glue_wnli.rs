//! Device-side WNLI GLUE contract. Requires a Vulkan device; the
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
#[ignore = "requires a Vulkan device and network or cached glue_wnli pack"]
fn glue_wnli_decodes_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::GlueWnli::load_on(&engine, &oads::GlueWnliConfig::default())?;
	assert_eq!(dataset.len(), 635);
	assert_eq!(dataset.labels().shape(), &[635]);
	assert_eq!(dataset.classes().len(), 2);
	assert!(dataset.text2().is_some());

	let item = dataset.get_item(0)?;
	assert_eq!(item.shape()[0], 2);
	let sample = dataset.get_sample(635 - 1)?;
	assert_eq!(sample.x.shape()[0], 2);
	Ok(())
}
