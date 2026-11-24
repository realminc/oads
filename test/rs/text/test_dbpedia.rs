//! Device-side dbpedia contract. Requires a Vulkan device; the
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
#[ignore = "requires a Vulkan device and network or cached dbpedia pack"]
fn dbpedia_loads_test_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Dbpedia::load_on(
		&engine,
		&oads::DbpediaConfig {
			split: oads::IdxSplit::Test,
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 70_000);
	assert_eq!(dataset.labels().shape(), &[70_000]);
	let labels = dataset.labels().read::<f32>()?;
	assert!(
		labels
			.iter()
			.all(|&v| v.fract() == 0.0 && (0.0..=13.0).contains(&v))
	);
	let item = dataset.get_item(0)?;
	assert_eq!(item.shape().len(), 1);
	Ok(())
}
