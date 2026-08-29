"""Tests for the generation backends.

Only the key lookup is tested, because it is the only part that is ours; the rest
is a thin wrapper over two SDKs. No network and no key needed.

The leak test earns its place: a TOML parse failure quotes the offending line,
which is the key itself, so folding that error into our message printed the
credential to the terminal and into any log that caught it. It happened once.
"""

import pytest

from claricyte.rag import providers

FAKE_KEY = "sk-proj-NOTAREALKEY-abcdef123456"


class ExplodingSecrets:
    """Stands in for st.secrets when the file will not parse."""

    def __contains__(self, key):
        raise ValueError(f"Error parsing secrets file: invalid literal '{FAKE_KEY}'")


def test_parse_failure_does_not_leak_the_key(monkeypatch, tmp_path):
    streamlit = pytest.importorskip("streamlit")
    secrets_file = tmp_path / "secrets.toml"
    secrets_file.write_text(f"OPENAI_API_KEY = {FAKE_KEY}\n", encoding="utf-8")
    monkeypatch.setattr(providers, "SECRETS_PATH", secrets_file)
    monkeypatch.setattr(streamlit, "secrets", ExplodingSecrets())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError) as raised:
        providers.api_key()
    message = str(raised.value)
    assert FAKE_KEY not in message
    assert "could not be parsed" in message
    assert "quoted" in message


def test_missing_key_says_so_plainly(monkeypatch, tmp_path):
    streamlit = pytest.importorskip("streamlit")
    monkeypatch.setattr(providers, "SECRETS_PATH", tmp_path / "absent.toml")
    monkeypatch.setattr(streamlit, "secrets", {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="No OPENAI_API_KEY"):
        providers.api_key()


def test_environment_variable_is_used_when_there_is_no_secrets_file(
    monkeypatch, tmp_path
):
    """The eval scripts run outside Streamlit, so the env var has to work alone."""
    streamlit = pytest.importorskip("streamlit")
    monkeypatch.setattr(providers, "SECRETS_PATH", tmp_path / "absent.toml")
    monkeypatch.setattr(streamlit, "secrets", {})
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    assert providers.api_key() == FAKE_KEY


def test_unknown_provider_name_is_rejected():
    with pytest.raises(ValueError, match="unknown provider"):
        providers.get_provider("gemini")
