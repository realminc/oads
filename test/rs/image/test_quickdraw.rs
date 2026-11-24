//! Device-side quickdraw contract. Requires a Vulkan device; the
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
#[ignore = "requires a Vulkan device and network or cached quickdraw pack"]
fn quickdraw_decodes_count_capped_rows() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Quickdraw::load_on(
		&engine,
		&oads::QuickdrawConfig {
			count: Some(1_000),
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 1_000);
	assert_eq!(dataset.images().shape(), &[1_000, 1, 28, 28]);
	assert_eq!(dataset.labels().shape(), &[1_000]);
	assert_eq!(dataset.classes(), 345);
	let item = dataset.get_item(0)?;
	assert_eq!(item.shape(), &[1, 28, 28]);
	Ok(())
}
