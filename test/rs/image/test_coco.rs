//! Device-side MS COCO detection contract. Requires a Vulkan device;
//! the pack is fetched once into `OA_DATA_DIR` (or `~/.oads`) and
//! reused.

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
#[ignore = "requires a Vulkan device and network or cached coco pack"]
fn coco_indexes_and_decodes_lazily() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Coco::load_on(
		&engine,
		&oads::CocoConfig {
			split: oads::CocoSplit::Validation,
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 5_000);
	assert_eq!(dataset.classes().len(), 80);
	assert_eq!(dataset.classes()[0], "person");

	let sample = dataset.get_sample(0)?;
	assert_eq!(sample.x.shape().len(), 3);
	assert_eq!(sample.x.shape()[0], 3);
	let y = sample.y.expect("detection labels");
	assert_eq!(y.shape()[1], 5);
	let rows = y.read::<f32>()?;
	assert!(rows.chunks(5).all(|b| b[2] >= b[0] && b[3] >= b[1]));
	assert!(
		rows
			.chunks(5)
			.all(|b| (0.0..80.0).contains(&b[4]) && b[4].fract() == 0.0)
	);
	Ok(())
}
