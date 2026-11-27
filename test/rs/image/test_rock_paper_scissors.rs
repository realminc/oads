//! Device-side rock-paper-scissors dataset contract. Requires a Vulkan
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
#[ignore = "requires a Vulkan device and network or cached rock_paper_scissors pack"]
fn rock_paper_scissors_indexes_and_decodes_lazily() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::RockPaperScissors::load_on(
		&engine,
		&oads::RockPaperScissorsConfig {
			split: oads::RockPaperScissorsSplit::Train,
			size: Some((224, 224)),
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 2_520);
	assert_eq!(dataset.classes(), &["paper", "rock", "scissors"]);
	assert_eq!(dataset.labels().shape(), &[2_520]);

	let labels = dataset.labels().read::<f32>()?;
	assert!(labels.iter().all(|label| (0.0..3.0).contains(label)));

	let item = dataset.get_item(0)?;
	assert_eq!(item.shape(), &[3, 224, 224]);
	let sample = dataset.get_sample(2_519)?;
	let batch = oa::data::collate(&[sample])?;
	assert_eq!(batch.x.shape(), &[1, 3, 224, 224]);
	assert_eq!(batch.y.as_ref().expect("labels").shape(), &[1, 1]);

	// The held-out archive indexes independently under the same config.
	let test = oads::RockPaperScissors::load_on(
		&engine,
		&oads::RockPaperScissorsConfig {
			split: oads::RockPaperScissorsSplit::Test,
			size: Some((224, 224)),
			..Default::default()
		},
	)?;
	assert_eq!(test.len(), 372);
	assert_eq!(test.classes(), &["paper", "rock", "scissors"]);
	assert_eq!(test.get_item(0)?.shape(), &[3, 224, 224]);
	Ok(())
}
