//! Device-side waves_yesno contract. Requires a Vulkan device; the
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
#[ignore = "requires a Vulkan device and network or cached waves_yesno pack"]
fn waves_yesno_decodes_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::WavesYesno::load_on(&engine, &oads::WavesYesnoConfig::default())?;
	assert_eq!(dataset.len(), 60);
	assert_eq!(dataset.labels().shape(), &[60, 8]);
	assert_eq!(dataset.lengths().shape(), &[60]);
	let wave = dataset.waveforms();
	assert_eq!(wave.shape()[0], 60);
	assert_eq!(wave.shape()[1], 1);

	let labels = dataset.labels().read::<f32>()?;
	assert!(labels.iter().all(|&v| v == 0.0 || v == 1.0));
	let lengths = dataset.lengths().read::<f32>()?;
	assert!(lengths.iter().all(|&n| n > 0.0));
	let item = dataset.get_item(0)?;
	assert_eq!(item.shape()[0], 1);
	let sample = dataset.get_sample(59)?;
	assert_eq!(sample.y.expect("sample carries a label").shape(), &[1, 8]);
	Ok(())
}
