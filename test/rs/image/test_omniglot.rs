//! Device-side Omniglot contract. Requires a Vulkan device; the pack
//! is fetched once into `OA_DATA_DIR` (or `~/.oads`) and reused.

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
#[ignore = "requires a Vulkan device and network or cached omniglot pack"]
fn omniglot_decodes_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Omniglot::load_on(&engine, &oads::OmniglotConfig::default())?;
	assert_eq!(dataset.len(), 19_280);
	assert_eq!(dataset.labels().shape(), &[19_280]);
	assert_eq!(dataset.classes().len(), 964);
	assert!(dataset.classes()[0].contains('/'));

	let item = dataset.get_item(0)?;
	assert_eq!(item.shape().len(), 3);
	let test = oads::Omniglot::load_on(
		&engine,
		&oads::OmniglotConfig {
			split: oads::OmniglotSplit::Test,
			..Default::default()
		},
	)?;
	assert_eq!(test.len(), 13_180);
	assert_eq!(test.classes().len(), 659);
	Ok(())
}
