//! Device-side VCTK dataset contract. Requires a Vulkan device;
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
#[ignore = "requires a Vulkan device and network or cached vctk pack"]
fn vctk_loads_speaker_labels_and_lazy_waveforms() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Vctk::load_on(&engine, &oads::VctkConfig::default())?;
	assert!(dataset.len() > 40_000);
	assert!(!dataset.speakers().is_empty());
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

	// Lazy waveform decode: clip length agrees with lengths.
	let lengths = dataset.lengths().read::<f32>()?;
	let wave = dataset.waveform(0)?;
	assert_eq!(wave.shape()[0], 1);
	assert_eq!(wave.shape()[1] as f32, lengths[0]);

	let sample = dataset.get_sample(1)?;
	assert_eq!(sample.x.shape()[0], 1);
	assert_eq!(sample.y.unwrap().shape(), &[1]);
	Ok(())
}
