//! Device-side coco_captions contract. Requires a Vulkan device; the
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
#[ignore = "requires a Vulkan device and network or cached coco_captions pack"]
fn coco_captions_indexes_and_decodes_lazily() -> oa::Result<()> {
	let engine = engine()?;
	let dataset = oads::CocoCaptions::load_on(
		&engine,
		&oads::CocoCaptionsConfig {
			split: oads::CocoCaptionsSplit::Validation,
			..Default::default()
		},
	)?;
	assert_eq!(dataset.len(), 25_014);
	assert_eq!(dataset.count(), 25_014);
	let caption = dataset.caption(0)?;
	assert_eq!(caption.shape().len(), 1);
	assert!(caption.shape()[0] > 0);
	let sample = dataset.get_sample(0)?;
	assert_eq!(sample.x.shape().len(), 3);
	assert_eq!(sample.x.shape()[0], 3);
	let y = sample.y.expect("caption bytes");
	assert_eq!(y.shape().len(), 1);
	Ok(())
}
