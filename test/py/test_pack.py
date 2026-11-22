"""Host-only pack resolution, fetch, and verification tests."""

import hashlib
import http.server
import threading
from pathlib import Path

import pytest

import oads
from oads import pack


def single_file_pack(source: Path, name: str, file_name: str, payload: bytes) -> dict:
	file = source / file_name
	file.write_bytes(payload)
	return {
		"name": name,
		"version": "0.0.1",
		"license": "test",
		"files": [
			{
				"name": file_name,
				"size": len(payload),
				"shake256": hashlib.shake_256(payload).hexdigest(32),
				"urls": [file.as_uri()],
			}
		],
	}


def test_hash_file_matches_known_shake256(tmp_path):
	path = tmp_path / "abc.bin"
	path.write_bytes(b"abc")
	assert (
		pack.hash_file(path)
		== "483366601360a8771c6863080cc4114d8db44530f8f1e1ee4f94ea37e78b5739"
	)


def test_ensure_pack_fetches_verifies_and_publishes(tmp_path):
	source = tmp_path / "source"
	source.mkdir()
	manifest = single_file_pack(source, "fetch", "payload.bin", b"pack bytes")
	root = tmp_path / "root"
	directory = pack.ensure_pack(manifest, root)
	assert directory == root / "fetch" / "0.0.1"
	assert (directory / "payload.bin").read_bytes() == b"pack bytes"


def test_ensure_pack_reuses_cached_members(tmp_path):
	source = tmp_path / "source"
	source.mkdir()
	manifest = single_file_pack(source, "cache", "payload.bin", b"pack bytes")
	root = tmp_path / "root"
	pack.ensure_pack(manifest, root)
	# A size-matching cache hit is accepted without touching the mirrors.
	manifest["files"][0]["urls"] = ["file:///definitely/missing.bin"]
	again = pack.ensure_pack(manifest, root)
	assert (again / "payload.bin").read_bytes() == b"pack bytes"


def test_ensure_pack_rejects_digest_mismatch(tmp_path):
	source = tmp_path / "source"
	source.mkdir()
	manifest = single_file_pack(source, "mismatch", "payload.bin", b"pack bytes")
	manifest["files"][0]["shake256"] = "0" * 64
	with pytest.raises(ValueError):
		pack.ensure_pack(manifest, tmp_path / "root")


def test_data_root_honors_oa_data_dir(tmp_path, monkeypatch):
	monkeypatch.setenv("OA_DATA_DIR", str(tmp_path))
	assert pack.data_root() == tmp_path


def test_ensure_pack_reports_progress(tmp_path):
	payload = bytes(index % 251 for index in range(2 * 1024 * 1024 + 7))
	source = tmp_path / "source"
	source.mkdir()
	manifest = single_file_pack(source, "progress", "blob.bin", payload)
	samples = []
	directory = pack.ensure_pack(
		manifest,
		tmp_path / "root",
		progress=lambda file, fetched, total: samples.append((file, fetched, total)),
	)
	assert samples, "progress samples expected"
	assert all(file == "blob.bin" and total == len(payload) for file, _, total in samples)
	assert [fetched for _, fetched, _ in samples] == sorted(
		fetched for _, fetched, _ in samples
	)
	assert samples[-1][1] == len(payload)
	assert (directory / "blob.bin").read_bytes() == payload


def test_ensure_pack_interrupt_keeps_partial_for_resume(tmp_path):
	payload = bytes(index % 253 for index in range(2 * 1024 * 1024 + 11))
	source = tmp_path / "source"
	source.mkdir()
	manifest = single_file_pack(source, "interrupt", "blob.bin", payload)
	root = tmp_path / "root"

	def abort(file, fetched, total):
		raise KeyboardInterrupt

	with pytest.raises(KeyboardInterrupt):
		pack.ensure_pack(manifest, root, progress=abort)
	directory = root / "interrupt" / "0.0.1"
	assert not (directory / "blob.bin").exists()
	assert (directory / "blob.bin.partial").exists()

	# The second run continues from the partial and publishes the member.
	finished = pack.ensure_pack(manifest, root)
	assert (finished / "blob.bin").read_bytes() == payload
	assert not (directory / "blob.bin.partial").exists()


class _RangeHandler(http.server.BaseHTTPRequestHandler):
	"""Serves one payload with HTTP Range support; records requests."""

	payload = b""
	ranges = []

	def do_GET(self):
		data = self.payload
		header = self.headers.get("Range")
		if header:
			start = int(header.removeprefix("bytes=").split("-")[0])
			type(self).ranges.append(header)
			self.send_response(206)
			self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
			data = data[start:]
		else:
			self.send_response(200)
		self.send_header("Content-Length", str(len(data)))
		self.end_headers()
		self.wfile.write(data)

	def log_message(self, *args):
		pass


@pytest.fixture
def range_server():
	server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
	thread = threading.Thread(target=server.serve_forever, daemon=True)
	thread.start()
	yield server
	server.shutdown()
	thread.join()


def test_ensure_pack_resumes_partial_over_http(tmp_path, range_server):
	# Seed a valid prefix as the partial; curl must continue from that
	# offset with a Range request instead of restarting the download.
	payload = bytes(index % 241 for index in range(400_000))
	prefix = 150_000
	_RangeHandler.payload = payload
	_RangeHandler.ranges = []
	manifest = {
		"name": "httpresume",
		"version": "0.0.1",
		"license": "test",
		"files": [
			{
				"name": "blob.bin",
				"size": len(payload),
				"shake256": hashlib.shake_256(payload).hexdigest(32),
				"urls": [f"http://127.0.0.1:{range_server.server_port}/blob.bin"],
			}
		],
	}
	root = tmp_path / "root"
	directory = root / "httpresume" / "0.0.1"
	directory.mkdir(parents=True)
	(directory / "blob.bin.partial").write_bytes(payload[:prefix])

	out = pack.ensure_pack(manifest, root)
	assert _RangeHandler.ranges == [f"bytes={prefix}-"]
	assert (out / "blob.bin").read_bytes() == payload
	assert not (directory / "blob.bin.partial").exists()


MANIFESTS = [
	oads.Mnist.PACK,
	oads.FashionMnist.PACK,
	oads.KMnist.PACK,
	oads.Cifar10.PACK,
	oads.text.tiny_shakespeare.PACK,
	oads.video.moving_mnist.PACK,
]


@pytest.mark.parametrize("manifest", MANIFESTS, ids=[m["name"] for m in MANIFESTS])
def test_registered_packs_carry_license_and_verified_files(manifest):
	assert manifest["license"]
	assert manifest["files"]
	for member in manifest["files"]:
		assert member["size"] > 0
		assert len(member["shake256"]) == 64
		assert member["urls"]
