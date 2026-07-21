"""Tests for backend helpers that don't need a live API."""

import pytest

from kiui.agent.backend import _is_fatal_api_error


class _StatusError(Exception):
    """Mimics an openai.APIStatusError instance carrying ``status_code``."""

    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_fatal_client_errors_are_not_retried(status):
    assert _is_fatal_api_error(_StatusError(status)) is True


@pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503])
def test_transient_errors_are_retried(status):
    assert _is_fatal_api_error(_StatusError(status)) is False


def test_errors_without_status_are_retried():
    # Connection errors / timeouts carry no status_code and must keep retrying.
    assert _is_fatal_api_error(Exception("connection reset")) is False
