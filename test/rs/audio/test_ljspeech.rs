//! Device-side LJSpeech dataset contract. Requires a Vulkan device;
//! the pack is fetched once into `OA_DATA_DIR` (or `~/.oads`) and reused.

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
#[ignore = "requires a Vulkan device and network or cached ljspeech pack"]
fn ljspeech_loads_transcripts_and_lazy_waveforms() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::LjSpeech::load_on(&engine, &oads::LjSpeechConfig::default())?;
	assert_eq!(dataset.len(), 13_100);
	assert_eq!(dataset.transcripts().shape()[0], 13_100);
	assert_eq!(dataset.audio_lengths().shape(), &[13_100]);

	// Lazy waveform decode: clip length agrees with audio_lengths.
	let lengths = dataset.audio_lengths().read::<f32>()?;
	let wave = dataset.waveform(0)?;
	assert_eq!(wave.shape()[0], 1);
	assert_eq!(wave.shape()[1] as f32, lengths[0]);

	let sample = dataset.get_sample(1)?;
	assert_eq!(sample.x.shape().len(), 1);
	assert_eq!(sample.y.unwrap().shape()[1] as f32, lengths[1]);
	Ok(())
}
