//! Device-side gtzan contract. Requires a Vulkan device; the
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
#[ignore = "requires a Vulkan device and network or cached gtzan pack"]
fn gtzan_decodes_clips_lazily() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Gtzan::load_on(&engine, &oads::GtzanConfig::default())?;
	assert_eq!(dataset.len(), 999);
	assert_eq!(dataset.genres().len(), 10);
	assert_eq!(dataset.genre_ids().shape(), &[999]);
	assert_eq!(dataset.lengths().shape(), &[999]);

	let waveform = dataset.get_item(0)?;
	assert_eq!(waveform.shape()[0], 1);
	assert_eq!(
		waveform.shape()[1],
		dataset.lengths().read_f32()?[0] as usize
	);
	let sample = dataset.get_sample(0)?;
	assert!(sample.y.is_some());
	Ok(())
}
