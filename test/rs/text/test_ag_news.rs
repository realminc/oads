//! Device-side AG News dataset contract. Requires a Vulkan device;
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
#[ignore = "requires a Vulkan device and network or cached ag_news pack"]
fn ag_news_loads_test_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::AgNews::load_on(
		&engine,
		&oads::AgNewsConfig {
			split: oads::IdxSplit::Test,
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 7_600);
	assert_eq!(dataset.text().shape(), &[7_600, 512]);
	assert_eq!(dataset.labels().shape(), &[7_600]);

	let labels = dataset.labels().read::<f32>()?;
	assert!(
		labels
			.iter()
			.all(|&v| v.fract() == 0.0 && (0.0..=3.0).contains(&v))
	);

	let first = dataset.get_sample(0)?;
	let second = dataset.get_sample(1)?;
	assert_eq!(first.x.shape(), &[512]);
	let batch = oa::data::collate(&[first, second])?;
	assert_eq!(batch.x.shape(), &[2, 512]);
	assert_eq!(batch.y.as_ref().expect("labels").shape(), &[2, 1]);
	Ok(())
}
