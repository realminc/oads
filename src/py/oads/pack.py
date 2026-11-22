"""Versioned, content-addressed dataset packs.

A manifest is a plain dict: name, version, and a files list where each member
records name, exact byte size, SHAKE256-256 hex digest, and mirror URLs.
ensure_pack materializes a manifest under a cache root — OA_DATA_DIR, then
~/.oads, then the platform cache — fetching missing members and
verifying every fetched byte against the digest before publishing it into
root/<name>/<version>. A cache hit checks the recorded byte size only;
delete the pack directory to force a refetch and full re-verification.

Transport runs in Rust through `oads._native`.
"""

from pathlib import Path

from . import _native
from ._progress import resolve


def data_root() -> Path:
	"""Resolve and create the dataset cache root."""
	return Path(_native.data_root())


def hash_file(path) -> str:
	"""Stream a file through SHAKE256-256 and return the hex digest."""
	return _native.hash_file(str(path))


def ensure_pack(manifest: dict, root, progress=None) -> Path:
	"""Materialize a pack manifest under root and return its directory.

	Downloads write `*.partial` members and resume from the recorded byte
	length; `progress` is a callable invoked as `(file, fetched, total)`
	or `True` for the default terminal renderer.
	"""
	return Path(_native.ensure_pack(manifest, str(root), resolve(progress)))
