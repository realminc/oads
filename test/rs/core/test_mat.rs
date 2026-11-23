use std::io::Write;

use flate2::{Compression, write::ZlibEncoder};

use super::*;

fn tag(ty: u32, data: &[u8]) -> Vec<u8> {
	let mut out = Vec::new();
	out.extend_from_slice(&ty.to_le_bytes());
	out.extend_from_slice(&(data.len() as u32).to_le_bytes());
	out.extend_from_slice(data);
	let pad = (8 - data.len() % 8) % 8;
	out.extend(std::iter::repeat_n(0, pad));
	out
}

fn matrix(name: &str, dims: &[i32], data_ty: u32, data: &[u8]) -> Vec<u8> {
	let mut body = Vec::new();
	// Array flags: miUINT32, mxDOUBLE_CLASS (6) or mxINT32 (12) in byte 0.
	let class = if data_ty == 9 { 6u8 } else { 12 };
	let mut flags = [0u8; 8];
	flags[0] = class;
	body.extend(tag(6, &flags));
	let dims_bytes: Vec<u8> = dims.iter().flat_map(|d| d.to_le_bytes()).collect();
	body.extend(tag(5, &dims_bytes));
	body.extend(tag(1, name.as_bytes()));
	body.extend(tag(data_ty, data));
	tag(14, &body)
}

fn mat_file(elements: &[Vec<u8>]) -> Vec<u8> {
	let mut out = vec![0u8; 128];
	out[..12].copy_from_slice(b"MATLAB 5.0 M");
	out[124..126].copy_from_slice(&0x0100u16.to_le_bytes());
	out[126..128].copy_from_slice(b"IM");
	for el in elements {
		out.extend_from_slice(el);
	}
	out
}

#[test]
fn reads_compressed_and_plain_numeric_arrays() {
	// trnid: int32 [1,3] inside a zlib-compressed element.
	let trnid: Vec<u8> = [10i32, 42, 9001]
		.iter()
		.flat_map(|v| v.to_le_bytes())
		.collect();
	let inner = matrix("trnid", &[1, 3], 5, &trnid);
	let mut enc = ZlibEncoder::new(Vec::new(), Compression::fast());
	enc.write_all(&inner).expect("compress");
	let compressed = enc.finish().expect("finish");
	let mut elements = vec![tag(15, &compressed)];
	// labels: double [1,3] as a plain top-level element.
	let labels: Vec<u8> = [1f64, 2.0, 77.0]
		.iter()
		.flat_map(|v| v.to_le_bytes())
		.collect();
	elements.push(matrix("labels", &[1, 3], 9, &labels));

	let file = mat_file(&elements);
	let arrays = read_arrays("test", &file).expect("read mat");
	assert_eq!(arrays.len(), 2);
	let trnid = &arrays["trnid"];
	assert_eq!(trnid.dims, vec![1, 3]);
	assert_eq!(trnid.values, vec![10.0, 42.0, 9001.0]);
	let labels = &arrays["labels"];
	assert_eq!(labels.values, vec![1.0, 2.0, 77.0]);
}

#[test]
fn rejects_non_v5_and_truncated_files() {
	assert!(read_arrays("test", b"not a mat file").is_err());
	let mut file = mat_file(&[]);
	file.extend_from_slice(&[9u8, 0, 0, 0, 100, 0, 0, 0]); // element claims 100B, absent
	assert!(read_arrays("test", &file).is_err());
}
