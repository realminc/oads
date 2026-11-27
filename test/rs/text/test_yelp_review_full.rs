//! Device-side yelp_review_full contract. Requires a Vulkan device; the
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
#[ignore = "requires a Vulkan device and network or cached yelp_review_full pack"]
fn yelp_review_full_loads_test_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::YelpReviewFull::load_on(
		&engine,
		&oads::YelpReviewFullConfig {
			split: oads::IdxSplit::Test,
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 50_000);
	assert_eq!(dataset.labels().shape(), &[50_000]);
	let labels = dataset.labels().read::<f32>()?;
	assert!(
		labels
			.iter()
			.all(|&v| v.fract() == 0.0 && (0.0..=4.0).contains(&v))
	);
	let first = dataset.get_sample(0)?;
	let second = dataset.get_sample(1)?;
	let batch = oa::data::collate(&[first, second])?;
	assert_eq!(batch.y.as_ref().expect("labels").shape(), &[2, 1]);
	Ok(())
}
