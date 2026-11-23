//! `oads._native` — the Rust decode and pack-transport boundary.
//!
//! `load` materializes a pack through the verified cache and decodes its
//! members on the host, returning `{field: {"shape": [...], "data": bytes}}`
//! plus optional `text` and `seq_len` keys. `data` is a native-order F32
//! payload; the Python layer wraps it in `array("f")` and uploads through
//! `oa.Matrix.new_from_f32`, so every decode path executes in Rust.

use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

mod error;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add("__version__", env!("CARGO_PKG_VERSION"))?;
	module.add_function(wrap_pyfunction!(data_root, module)?)?;
	module.add_function(wrap_pyfunction!(hash_file, module)?)?;
	module.add_function(wrap_pyfunction!(ensure_pack, module)?)?;
	module.add_function(wrap_pyfunction!(pack_manifest, module)?)?;
	module.add_function(wrap_pyfunction!(load, module)?)?;
	Ok(())
}

/// Resolve the dataset cache root (`OA_DATA_DIR`, `~/.oads`, XDG cache).
#[pyfunction]
fn data_root() -> PyResult<String> {
	let root = oads::data_root().map_err(error::python_error)?;
	Ok(root.to_string_lossy().into_owned())
}

/// SHAKE256-256 hex digest of a file.
#[pyfunction]
fn hash_file(path: PathBuf) -> PyResult<String> {
	let hash = oads::hash_file(&path).map_err(error::python_error)?;
	Ok(hash.to_hex())
}

/// Materialize a manifest dict under `root` and return the pack directory.
///
/// `progress` is an optional callable invoked as `(file, fetched, total)`
/// while members download; raising inside it cancels the fetch.
#[pyfunction(signature = (manifest, root, progress = None))]
fn ensure_pack(
	py: Python<'_>,
	manifest: &Bound<'_, PyDict>,
	root: PathBuf,
	progress: Option<Py<PyAny>>,
) -> PyResult<String> {
	let pack = pack_from_dict(manifest)?;
	let mut callback_error: Option<PyErr> = None;
	let result = py.detach(|| {
		oads::ensure_pack_with_progress(&pack, &root, &mut |sample| {
			report(progress.as_ref(), &mut callback_error, sample)
		})
	});
	if let Some(error) = callback_error {
		return Err(error);
	}
	let directory = result.map_err(error::python_error)?;
	Ok(directory.to_string_lossy().into_owned())
}

/// Return the compiled manifest for a registered pack as a dict.
#[pyfunction]
fn pack_manifest(py: Python<'_>, name: &str) -> PyResult<Py<PyDict>> {
	let pack = oads::decode::manifest(name).map_err(error::python_error)?;
	Ok(manifest_to_dict(py, pack)?.unbind())
}

/// Decode a registered pack on the host.
///
/// Returns a dict of matrix fields (`{"shape": [...], "data": f32 bytes}`)
/// plus `text` (corpus packs) and `seq_len` (sequence packs) when present.
#[pyfunction(signature = (name, split = "train", normalize = true, count = None, cache_dir = None, progress = None))]
fn load(
	py: Python<'_>,
	name: &str,
	split: &str,
	normalize: bool,
	count: Option<usize>,
	cache_dir: Option<PathBuf>,
	progress: Option<Py<PyAny>>,
) -> PyResult<Py<PyDict>> {
	let split = match split {
		"train" => oads::IdxSplit::Train,
		"test" => oads::IdxSplit::Test,
		"validation" => oads::IdxSplit::Validation,
		other => {
			return Err(PyValueError::new_err(format!(
				"unknown split '{other}'; expected 'train', 'test', or 'validation'"
			)));
		}
	};
	let config = oads::decode::DecodeConfig {
		split,
		normalize,
		count,
		cache_dir,
	};
	let mut callback_error: Option<PyErr> = None;
	let decoded = py
		.detach(|| {
			oads::decode::load_with_progress(name, &config, &mut |sample| {
				report(progress.as_ref(), &mut callback_error, sample)
			})
		})
		.map_err(error::python_error)?;
	if let Some(error) = callback_error {
		return Err(error);
	}

	let result = PyDict::new(py);
	for matrix in &decoded.matrices {
		let field = PyDict::new(py);
		field.set_item("shape", matrix.shape.clone())?;
		field.set_item("data", PyBytes::new(py, f32_bytes(&matrix.values)))?;
		result.set_item(matrix.name, field)?;
	}
	if let Some(text) = &decoded.text {
		result.set_item("text", text)?;
	}
	if decoded.seq_len > 0 {
		result.set_item("seq_len", decoded.seq_len)?;
	}
	Ok(result.unbind())
}

/// Invoke the Python progress callable with one fetch sample.
///
/// Returns `false` once a callback raised — the first exception is stored in
/// `error` and the fetch unwinds as `Cancelled` so the caller can surface
/// the original exception (for example `KeyboardInterrupt`).
fn report(
	progress: Option<&Py<PyAny>>,
	error: &mut Option<PyErr>,
	sample: oads::FetchProgress,
) -> bool {
	if error.is_some() {
		return false;
	}
	let Some(callback) = progress else {
		return true;
	};
	match Python::attach(|py| callback.call1(py, (sample.file, sample.fetched, sample.total))) {
		Ok(_) => true,
		Err(raised) => {
			*error = Some(raised);
			false
		}
	}
}

fn f32_bytes(values: &[f32]) -> &[u8] {
	// SAFETY: u8 has no alignment or validity requirements; the returned
	// slice covers exactly the f32 payload bytes in native order.
	unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}

fn pack_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<oads::Pack> {
	fn leaked(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<&'static str> {
		let value = dict
			.get_item(key)?
			.ok_or_else(|| PyValueError::new_err(format!("pack manifest lacks '{key}'")))?;
		let text = value.extract::<String>()?;
		Ok(Box::leak(text.into_boxed_str()))
	}
	let files_value = dict
		.get_item("files")?
		.ok_or_else(|| PyValueError::new_err("pack manifest lacks 'files'"))?;
	let files_list = files_value.cast::<PyList>()?;
	let mut files = Vec::with_capacity(files_list.len());
	for item in files_list.iter() {
		let file = item.cast::<PyDict>()?;
		let size = file
			.get_item("size")?
			.ok_or_else(|| PyValueError::new_err("pack file lacks 'size'"))?
			.extract::<u64>()?;
		let urls_value = file
			.get_item("urls")?
			.ok_or_else(|| PyValueError::new_err("pack file lacks 'urls'"))?;
		let urls_list = urls_value.cast::<PyList>()?;
		let mut urls = Vec::with_capacity(urls_list.len());
		for url in urls_list.iter() {
			urls.push(Box::leak(url.extract::<String>()?.into_boxed_str()) as &'static str);
		}
		files.push(oads::PackFile {
			name: leaked(file, "name")?,
			size,
			shake256: leaked(file, "shake256")?,
			urls: Box::leak(urls.into_boxed_slice()),
		});
	}
	Ok(oads::Pack {
		name: leaked(dict, "name")?,
		version: leaked(dict, "version")?,
		license: leaked(dict, "license")?,
		files: Box::leak(files.into_boxed_slice()),
	})
}

fn manifest_to_dict<'py>(
	py: Python<'py>,
	pack: &'static oads::Pack,
) -> PyResult<Bound<'py, PyDict>> {
	let dict = PyDict::new(py);
	dict.set_item("name", pack.name)?;
	dict.set_item("version", pack.version)?;
	dict.set_item("license", pack.license)?;
	let files = PyList::empty(py);
	for file in pack.files {
		let member = PyDict::new(py);
		member.set_item("name", file.name)?;
		member.set_item("size", file.size)?;
		member.set_item("shake256", file.shake256)?;
		let urls = PyList::empty(py);
		for url in file.urls {
			urls.append(url)?;
		}
		member.set_item("urls", urls)?;
		files.append(member)?;
	}
	dict.set_item("files", files)?;
	Ok(dict)
}
