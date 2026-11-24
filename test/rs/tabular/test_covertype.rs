//! Device-side covertype contract. Requires a Vulkan device; the
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
#[ignore = "requires a Vulkan device and network or cached covertype pack"]
fn covertype_loads_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Covertype::load_on(&engine, &oads::CovertypeConfig::default())?;
	assert_eq!(dataset.len(), 581_012);
	assert_eq!(dataset.features().shape(), &[581_012, 54]);
	assert_eq!(dataset.labels().shape(), &[581_012]);
	let labels = dataset.labels().read::<f32>()?;
	assert!(
		labels
			.iter()
			.all(|&v| v.fract() == 0.0 && (0.0..=6.0).contains(&v))
	);
	let item = dataset.get_item(0)?;
	assert_eq!(item.shape(), &[54]);
	Ok(())
}
