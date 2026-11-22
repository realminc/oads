//! Device-side MAESTRO window dataset contract. Requires a Vulkan device;
//! the pack is fetched once into `OA_DATA_DIR` (or the platform cache) and
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
#[ignore = "requires a Vulkan device and network or cached maestro pack"]
fn maestro_loads_windows_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::Maestro::load_on(&engine, &oads::MaestroConfig::default())?;
	assert!(dataset.len() > 0);
	assert_eq!(dataset.notes().shape()[1], 3);
	assert_eq!(dataset.seq_len(), oads::audio::DEFAULT_SEQ_LEN);
	assert_eq!(dataset.split(), oads::MaestroSplit::Train);

	// Items are [seq_len, 3] windows; samples pair them with the
	// one-event-shifted window as the next-token target.
	let seq_len = dataset.seq_len() as i64;
	let first = dataset.get_sample(0)?;
	assert_eq!(first.x.shape(), &[seq_len as usize, 3]);
	let y = first.y.as_ref().expect("target").read::<f32>()?;
	let x = first.x.read::<f32>()?;
	for i in 0..seq_len as usize - 1 {
		assert_eq!(x[3 * (i + 1)], y[3 * i], "shifted pitch mismatch at {i}");
	}
	let batch = oa::data::collate(&[first, dataset.get_sample(1)?])?;
	assert_eq!(batch.x.shape(), &[2, seq_len as usize, 3]);

	// Events stay inside MIDI's pitch and nonnegative timing contract.
	let notes = dataset.notes().read::<f32>()?;
	assert!(
		notes
			.chunks(3)
			.all(|event| { (0.0..=127.0).contains(&event[0]) && event[1] >= 0.0 && event[2] >= 0.0 })
	);
	Ok(())
}
