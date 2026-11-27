//! Device-side PatchCamelyon contract. Requires a Vulkan device; the
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
#[ignore = "requires a Vulkan device and network or cached patch_camelyon pack"]
fn patch_camelyon_decodes_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::PatchCamelyon::load_on(
		&engine,
		&oads::PatchCamelyonConfig {
			split: oads::PatchCamelyonSplit::Validation,
			count: Some(1_000),
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 1_000);
	assert_eq!(dataset.images().shape(), &[1_000, 3, 96, 96]);
	assert_eq!(dataset.labels().shape(), &[1_000]);
	let labels = dataset.labels().read::<f32>()?;
	assert!(
		labels
			.iter()
			.all(|l| l.fract() == 0.0 && (0.0..2.0).contains(l))
	);
	let item = dataset.get_item(0)?;
	assert_eq!(item.shape(), &[3, 96, 96]);
	Ok(())
}
