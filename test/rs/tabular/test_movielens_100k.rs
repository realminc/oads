//! Device-side MovieLens 100k dataset contract. Requires a Vulkan
//! device; the pack is fetched once into `OA_DATA_DIR` (or `~/.oads`)
//! and reused.

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
#[ignore = "requires a Vulkan device and network or cached movielens_100k pack"]
fn movielens_100k_loads_test_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Movielens100k::load_on(
		&engine,
		&oads::Movielens100kConfig {
			split: oads::Movielens100kSplit::Test,
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 20_000);
	assert_eq!(dataset.split(), oads::Movielens100kSplit::Test);
	assert_eq!(dataset.features().shape(), &[20_000, 3]);
	assert_eq!(dataset.ratings().shape(), &[20_000]);

	// Ratings are integral in 1..=5.
	let ratings = dataset.ratings().read::<f32>()?;
	assert!(
		ratings
			.iter()
			.all(|v| *v == v.trunc() && (1.0..=5.0).contains(v))
	);

	let batch = oa::data::collate(&[dataset.get_sample(0)?, dataset.get_sample(1)?])?;
	assert_eq!(batch.x.shape(), &[2, 3]);
	Ok(())
}
