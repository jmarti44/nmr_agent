import pytest


@pytest.fixture(autouse=True)
def _fresh_usage_guard():
    """The abuse-control counters are process-wide; every test starts with empty ones."""
    from src.api.guards import get_guard

    get_guard.cache_clear()
    yield
    get_guard.cache_clear()
