//! Device-side mini Speech Commands dataset contract. Requires a Vulkan
//! device; the pack is fetched once into `OA_DATA_DIR` (or the platform
//! cache) and reused.

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
#[ignore = "requires a Vulkan device and network or cached mini_speech_commands pack"]
fn mini_speech_commands_loads_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset =
		oads::MiniSpeechCommands::load_on(&engine, &oads::MiniSpeechCommandsConfig::default())?;
	assert_eq!(dataset.len(), 8_000);
	assert_eq!(dataset.samples().shape(), &[8_000, 1, 16_000]);
	assert_eq!(dataset.labels().shape(), &[8_000]);

	// Labels are integral class indices in 0..=7 covering every command.
	let labels = dataset.labels().read::<f32>()?;
	assert!(
		labels
			.iter()
			.all(|label| label.fract() == 0.0 && (0.0..8.0).contains(label))
	);
	for command in 0..oads::audio::COMMANDS.len() {
		assert!(labels.contains(&(command as f32)));
	}

	// Per-sample waveform slices collate into [B, 1, 16000] batches.
	let first = dataset.get_sample(0)?;
	let second = dataset.get_sample(1)?;
	assert_eq!(first.x.shape(), &[1, 16_000]);
	assert_eq!(first.y.as_ref().expect("label").shape(), &[1]);
	let batch = oa::data::collate(&[first, second])?;
	assert_eq!(batch.x.shape(), &[2, 1, 16_000]);

	// Normalized PCM lands in [-1, 1] and carries real signal energy.
	let samples = dataset.samples().read::<f32>()?;
	assert!(samples.iter().all(|&v| (-1.0..=1.0).contains(&v)));
	assert!(samples.iter().any(|&v| v.abs() > 0.01));
	Ok(())
}
