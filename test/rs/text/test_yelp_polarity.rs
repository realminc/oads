//! Device-side Yelp Polarity dataset contract. Requires a Vulkan device;
//! the pack is fetched once into `OA_DATA_DIR` (or `~/.oads`) and reused.

use oa::data::Dataset;

fn engine() -> oa::Result<oa::Runtime> {
	let engine = oa::Runtime::builder()
		.devices(oa::DeviceSelection::Automatic)
		.build()?;
	oa::Runtime::set_default(engine.clone());
	Ok(engine)
}

#[test]
#[ignore = "requires a Vulkan device and network or cached yelp_polarity pack"]
fn yelp_polarity_loads_test_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::YelpPolarity::load_on(
		&engine,
		&oads::YelpPolarityConfig {
			split: oads::IdxSplit::Test,
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 38_000);
	assert_eq!(dataset.text().shape(), &[38_000, 512]);
	assert_eq!(dataset.labels().shape(), &[38_000]);

	let labels = dataset.labels().read::<f32>()?;
	assert!(labels.iter().all(|&v| v == 0.0 || v == 1.0));

	let first = dataset.get_sample(0)?;
	let second = dataset.get_sample(1)?;
	assert_eq!(first.x.shape(), &[512]);
	let batch = oa::data::collate(&[first, second])?;
	assert_eq!(batch.x.shape(), &[2, 512]);
	Ok(())
}
