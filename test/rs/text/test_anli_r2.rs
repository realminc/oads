//! Device-side anli_r2 contract. Requires a Vulkan device; the
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
#[ignore = "requires a Vulkan device and network or cached anli_r2 pack"]
fn anli_r2_decodes_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::AnliR2::load_on(&engine, &oads::AnliR2Config::default())?;
	assert_eq!(dataset.len(), 45460);
	assert_eq!(dataset.labels().expect("labels").shape(), &[45460]);
	assert_eq!(dataset.classes().len(), 3);
	let item = dataset.get_item(0)?;
	assert_eq!(item.shape()[0], 1);
	Ok(())
}
