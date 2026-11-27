"""Shared fixtures: one engine and the pack cache root for the session."""

import oa
import pytest
from oads import pack


@pytest.fixture(scope="session")
def engine():
	return oa.Runtime(logging=oa.LogOptions().console_output(False))


@pytest.fixture(scope="session")
def cache_root():
	return pack.data_root()
