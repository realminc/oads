//! Device-side SQuAD v2 contract. Requires a Vulkan device; the
//! pack is fetched once into `OA_DATA_DIR` (or `~/.oads`) and reused.

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
#[ignore = "requires a Vulkan device and network or cached squad_v2 pack"]
fn squad_v2_decodes_split_and_collates() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::SquadV2::load_on(&engine, &oads::SquadV2Config::default())?;
	assert_eq!(dataset.len(), 130_319);
	assert_eq!(dataset.question_offsets().shape(), &[130_319]);
	assert_eq!(dataset.question_lengths().shape(), &[130_319]);
	assert_eq!(dataset.context_index().shape(), &[130_319]);
	assert_eq!(dataset.context_offsets().shape().len(), 1);
	assert_eq!(dataset.answer_start().shape(), &[130_319]);
	assert_eq!(dataset.is_impossible().shape(), &[130_319]);
	assert_eq!(dataset.answer_offsets().shape(), &[130_319]);
	assert!(dataset.context_offsets().shape()[0] < 130_319);

	let starts = dataset.answer_start().read::<f32>()?;
	assert!(starts.iter().any(|&s| s == -1.0));
	let item = dataset.get_item(0)?;
	assert_eq!(item.shape().len(), 1);
	let sample = dataset.get_sample(130_318)?;
	assert_eq!(sample.y.expect("sample carries a label").shape(), &[1]);
	Ok(())
}
