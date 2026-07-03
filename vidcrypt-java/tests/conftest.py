"""Shared pytest configuration for Vidcrypt test suite."""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", 'slow: marks tests as slow (deselect with \'-m "not slow"\')'
    )


@pytest.fixture(scope="session")
def java_project_root():
    """Path to the Java project root."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(test_dir)
