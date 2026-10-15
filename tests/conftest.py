import pytest


@pytest.fixture(autouse=True)
def _fresh_engine_cache():
    """The OCR model cache is process-wide; keep tests independent of each other."""
    from packet.backends import clear_engine_cache

    clear_engine_cache()
    yield
    clear_engine_cache()
