import importlib
import pytest
from config import settings

def test_all_keys_present(monkeypatch):
    for k in ["API_KEY", "MILVUS_URL", "EMBED_MODEL", "EMBED_BASE_URL", "MODEL", "BASE_URL"]:
        monkeypatch.setenv(k, "dummy")
    importlib.reload(settings)
    assert all([settings.API_KEY, settings.MILVUS_URL, settings.EMBED_MODEL,
                settings.EMBED_BASE_URL, settings.MODEL, settings.BASE_URL])

def test_missing_key_raises(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", None)
    with pytest.raises(AssertionError):
        settings.assert_env()