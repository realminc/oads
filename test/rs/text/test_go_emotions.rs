//! Device-side go_emotions contract. Requires a Vulkan device; the
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
#[ignore = "requires a Vulkan device and network or cached go_emotions pack"]
fn go_emotions_loads_split_and_multi_hot_labels() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::GoEmotions::load_on(
		&engine,
		&oads::GoEmotionsConfig {
			split: oads::GoEmotionsSplit::Validation,
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 5_426);
	assert_eq!(dataset.labels().shape(), &[5_426, 28]);
	let labels = dataset.labels().read::<f32>()?;
	assert!(labels.iter().all(|&v| v == 0.0 || v == 1.0));
	assert!(labels.chunks(28).all(|row| row.iter().sum::<f32>() >= 1.0));
	Ok(())
}
