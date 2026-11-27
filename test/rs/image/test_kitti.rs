//! Device-side KITTI detection contract. Requires a Vulkan device;
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
#[ignore = "requires a Vulkan device and network or cached kitti pack"]
fn kitti_indexes_and_decodes_lazily() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Kitti::load_on(&engine, &oads::KittiConfig::default())?;
	assert_eq!(dataset.len(), 7_481);
	assert!(dataset.classes().contains(&"Car".to_owned()));
	assert!(!dataset.classes().contains(&"DontCare".to_owned()));

	let sample = dataset.get_sample(0)?;
	assert_eq!(sample.x.shape().len(), 3);
	assert_eq!(sample.x.shape()[0], 3);
	let y = sample.y.expect("detection labels");
	assert_eq!(y.shape()[1], 5);
	let rows = y.read::<f32>()?;
	assert!(rows.chunks(5).all(|b| b[2] >= b[0] && b[3] >= b[1]));

	// Unlabeled test images index with empty box slices.
	let test = oads::Kitti::load_on(
		&engine,
		&oads::KittiConfig {
			split: oads::KittiSplit::Test,
			..Default::default()
		},
	)?;
	assert_eq!(test.len(), 7_518);
	assert_eq!(
		test.get_sample(0)?.y.expect("detection labels").shape(),
		&[0, 5]
	);
	Ok(())
}
