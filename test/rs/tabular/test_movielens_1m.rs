//! Device-side MovieLens 1M dataset contract. Requires a Vulkan device;
//! the pack is fetched once into `OA_DATA_DIR` (or `~/.oads`) and reused.

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
#[ignore = "requires a Vulkan device and network or cached movielens_1m pack"]
fn movielens_1m_loads_all_rows_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Movielens1m::load_on(&engine, &oads::Movielens1mConfig::default())?;
	assert_eq!(dataset.len(), 1_000_209);
	assert_eq!(dataset.features().shape(), &[1_000_209, 3]);
	assert_eq!(dataset.ratings().shape(), &[1_000_209]);

	let batch = oa::data::collate(&[dataset.get_sample(0)?, dataset.get_sample(1)?])?;
	assert_eq!(batch.x.shape(), &[2, 3]);
	Ok(())
}
