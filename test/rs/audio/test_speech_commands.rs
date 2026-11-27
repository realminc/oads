//! Device-side Speech Commands v0.02 dataset contract. Requires a Vulkan
//! device; the pack is fetched once into `OA_DATA_DIR` (or `~/.oads`) and
//! reused.

use oa::data::Dataset;

fn engine() -> oa::Result<oa::Runtime> {
	let engine = oa::Runtime::builder()
		.devices(oa::DeviceSelection::Automatic)
		.build()?;
	oa::Runtime::set_default(engine.clone());
	Ok(engine)
}

#[test]
#[ignore = "requires a Vulkan device and network or cached speech_commands pack"]
fn speech_commands_loads_test_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::SpeechCommands::load_on(
		&engine,
		&oads::SpeechCommandsConfig {
			split: oads::SpeechCommandsSplit::Test,
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 11_005);
	assert_eq!(dataset.samples().shape(), &[11_005, 1, 16_000]);
	assert_eq!(dataset.labels().shape(), &[11_005]);

	let labels = dataset.labels().read::<f32>()?;
	assert!(
		labels
			.iter()
			.all(|&v| v.fract() == 0.0 && (0.0..=34.0).contains(&v))
	);

	let first = dataset.get_sample(0)?;
	let second = dataset.get_sample(1)?;
	assert_eq!(first.x.shape(), &[1, 16_000]);
	let batch = oa::data::collate(&[first, second])?;
	assert_eq!(batch.x.shape(), &[2, 1, 16_000]);

	let samples = dataset.samples().read::<f32>()?;
	assert!(samples.iter().all(|&v| (-1.0..=1.0).contains(&v)));
	Ok(())
}
