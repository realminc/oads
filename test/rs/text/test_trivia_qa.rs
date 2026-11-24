//! Device-side trivia_qa contract. Requires a Vulkan device; the pack is
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
#[ignore = "requires a Vulkan device and network or cached trivia_qa pack"]
fn trivia_qa_decodes_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::TriviaQa::load_on(&engine, &oads::TriviaQaConfig::default())?;
	assert_eq!(dataset.len(), 61_888);
	assert_eq!(dataset.field(0)?.shape()[0], 61_888);
	assert!(dataset.labels().is_none());
	let item = dataset.get_item(0)?;
	assert_eq!(item.shape()[0], 1);
	Ok(())
}
