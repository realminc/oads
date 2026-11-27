//! Device-side horses-or-humans dataset contract. Requires a Vulkan
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
#[ignore = "requires a Vulkan device and network or cached horses_or_humans pack"]
fn horses_or_humans_indexes_and_decodes_lazily() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::HorsesOrHumans::load_on(
		&engine,
		&oads::HorsesOrHumansConfig {
			split: oads::HorsesOrHumansSplit::Train,
			size: Some((224, 224)),
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 1_027);
	assert_eq!(dataset.classes(), &["horses", "humans"]);
	assert_eq!(dataset.labels().shape(), &[1_027]);

	let labels = dataset.labels().read::<f32>()?;
	assert!(labels.iter().all(|label| *label == 0.0 || *label == 1.0));

	let item = dataset.get_item(0)?;
	assert_eq!(item.shape(), &[3, 224, 224]);
	let sample = dataset.get_sample(1_026)?;
	let batch = oa::data::collate(&[sample])?;
	assert_eq!(batch.x.shape(), &[1, 3, 224, 224]);
	assert_eq!(batch.y.as_ref().expect("labels").shape(), &[1, 1]);

	let validation = oads::HorsesOrHumans::load_on(
		&engine,
		&oads::HorsesOrHumansConfig {
			split: oads::HorsesOrHumansSplit::Validation,
			size: Some((224, 224)),
			..Default::default()
		},
	)?;
	assert_eq!(validation.len(), 256);
	assert_eq!(validation.classes(), &["horses", "humans"]);
	assert_eq!(validation.get_item(0)?.shape(), &[3, 224, 224]);
	Ok(())
}
