//! Device-side small NORB contract. Requires a Vulkan device; the
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
#[ignore = "requires a Vulkan device and network or cached smallnorb pack"]
fn smallnorb_decodes_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Smallnorb::load_on(&engine, &oads::SmallnorbConfig::default())?;
	assert_eq!(dataset.len(), 24_300);
	assert_eq!(dataset.images().shape()[1..], [2, 96, 96]);
	assert_eq!(dataset.labels().shape(), &[24_300]);
	assert_eq!(dataset.info().shape(), &[24_300, 4]);
	assert_eq!(oads::image::smallnorb::CLASSES.len(), 5);

	let labels = dataset.labels().read::<f32>()?;
	assert!(
		labels
			.iter()
			.all(|&v| v == v.trunc() && (0.0..5.0).contains(&v))
	);
	let item = dataset.get_item(0)?;
	assert_eq!(item.shape(), &[2, 96, 96]);
	Ok(())
}
