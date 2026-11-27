//! Device-side LibriSpeech dataset contract. Requires a Vulkan
//! device; the pack is fetched once into `OA_DATA_DIR` (or `~/.oads`)
//! and reused.

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
#[ignore = "requires a Vulkan device and network or cached librispeech pack"]
fn librispeech_loads_speakers_and_lazy_waveforms() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Librispeech::load_on(&engine, &oads::LibrispeechConfig::default())?;
	assert_eq!(dataset.len(), 2_703);
	assert_eq!(dataset.speakers().len(), 40);
	let len = usize::try_from(dataset.len()).unwrap_or(usize::MAX);
	assert_eq!(dataset.speaker_ids().shape()[0], len);
	assert_eq!(dataset.lengths().shape()[0], len);
	assert_eq!(dataset.transcripts().shape()[0], len);

	// Speaker ids index into the sorted speaker list.
	let ids = dataset.speaker_ids().read::<f32>()?;
	let max = dataset.speakers().len() as f32;
	assert!(
		ids
			.iter()
			.all(|&id| id == id.trunc() && (0.0..max).contains(&id))
	);

	// Lazy decode: waveform length matches the indexed sample count.
	let lengths = dataset.lengths().read::<f32>()?;
	let wave = dataset.get_item(0)?;
	assert_eq!(wave.shape()[0], 1);
	assert_eq!(wave.shape()[1] as f32, lengths[0]);

	let sample = dataset.get_sample(2_702)?;
	assert_eq!(sample.x.shape()[0], 1);
	let y = sample.y.as_ref().expect("speaker label");
	assert_eq!(y.shape(), &[1]);
	Ok(())
}
