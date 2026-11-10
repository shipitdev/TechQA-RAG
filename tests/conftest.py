"""Offline checks must never export local test prompts to an account."""

import pytest


@pytest.fixture(autouse=True)
def disable_external_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
