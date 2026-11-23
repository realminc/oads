//! Device-side NSynth dataset contract. Requires a Vulkan device;
//! the pack is fetched once into `OA_DATA_DIR` (or `~/.oads`) and
//! reused.

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
#[ignore = "requires a Vulkan device and network or cached nsynth pack"]
fn nsynth_loads_families_and_lazy_waveforms() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Nsynth::load_on(&engine, &oads::NsynthConfig::default())?;
	assert_eq!(dataset.len(), 4_096);
	assert_eq!(dataset.families().len(), 11);
	let len = usize::try_from(dataset.len()).unwrap_or(usize::MAX);
	assert_eq!(dataset.labels().shape()[0], len);
	assert_eq!(dataset.pitches().shape()[0], len);
	assert_eq!(dataset.velocities().shape()[0], len);

	// Family labels index into the 11-class table.
	let labels = dataset.labels().read::<f32>()?;
	assert!(
		labels
			.iter()
			.all(|&id| id == id.trunc() && (0.0..11.0).contains(&id))
	);

	// Every note is 4 seconds at 16,000 Hz.
	let lengths = dataset.lengths().read::<f32>()?;
	assert!(lengths.iter().all(|&samples| samples == 64_000.0));
	let wave = dataset.get_item(0)?;
	assert_eq!(wave.shape(), &[1, 64_000]);

	let sample = dataset.get_sample(4_095)?;
	assert_eq!(sample.x.shape()[0], 1);
	let y = sample.y.as_ref().expect("family label");
	assert_eq!(y.shape(), &[1]);
	Ok(())
}
