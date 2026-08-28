"""Shared pytest configuration.

Adds the ``live`` marker: tests marked ``live`` need a running mlx-serve
instance (and, for video, ffmpeg) and are skipped unless ``--run-live`` is
passed.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-live", action="store_true", default=False,
                     help="run @pytest.mark.live tests (need a running mlx-serve)")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live: needs a running mlx-serve instance")


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    if config.getoption("--run-live"):
        return
    skip = pytest.mark.skip(reason="live test (pass --run-live to enable)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)