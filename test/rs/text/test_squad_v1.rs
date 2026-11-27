//! Device-side squad_v1 contract. Requires a Vulkan device; the
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
#[ignore = "requires a Vulkan device and network or cached squad_v1 pack"]
fn squad_v1_decodes_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::SquadV1::load_on(
		&engine,
		&oads::SquadV1Config {
			split: oads::SquadV1Split::Validation,
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 10_570);
	let item = dataset.get_item(0)?;
	assert_eq!(item.shape().len(), 1);
	let sample = dataset.get_sample(0)?;
	assert_eq!(sample.y.expect("answer start").shape(), &[1]);
	Ok(())
}
