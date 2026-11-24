//! Device-side PASCAL VOC 2007 detection contract. Requires a Vulkan
//! device; the pack is fetched once into `OA_DATA_DIR` (or `~/.oads`)
//! and reused.

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
#[ignore = "requires a Vulkan device and network or cached voc2007 pack"]
fn voc2007_indexes_and_decodes_lazily() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Voc2007::load_on(&engine, &oads::Voc2007Config::default())?;
	assert_eq!(dataset.len(), 2_501);
	assert_eq!(dataset.classes().len(), 20);
	assert_eq!(dataset.classes()[0], "aeroplane");

	let sample = dataset.get_sample(0)?;
	assert_eq!(sample.x.shape().len(), 3);
	assert_eq!(sample.x.shape()[0], 3);
	let y_mat = sample.y.expect("detection labels");
	assert_eq!(y_mat.shape()[1], 5);
	// Box columns are [x1, y1, x2, y2, class] in pixel space.
	let y = y_mat.read::<f32>()?;
	assert!(y.chunks(5).all(|b| b[2] >= b[0] && b[3] >= b[1]));
	assert!(
		y.chunks(5)
			.all(|b| (0.0..20.0).contains(&b[4]) && b[4].fract() == 0.0)
	);

	// Every image in the official train list indexes; test split is larger.
	let last = dataset.get_sample(2_500)?;
	assert_eq!(last.y.expect("detection labels").shape()[1], 5);
	Ok(())
}
