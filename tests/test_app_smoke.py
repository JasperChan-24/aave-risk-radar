from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from aave_risk_monitor.app.demo import DEMO_WARNING

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_default_demo_renders_offline_without_exceptions(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ALCHEMY_RPC_URL", raising=False)
    monkeypatch.delenv("ETHEREUM_RPC_URL", raising=False)

    app = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "🌊 Aave Whale Asset-Level Risk Monitor"
    assert app.sidebar.radio[0].value == "English"
    assert app.sidebar.radio[1].value == "Fixed demo snapshot"
    assert len(app.dataframe) >= 1
    assert any(DEMO_WARNING in warning.value for warning in app.warning)


def test_streamlit_live_mode_reports_invalid_settings_without_uncaught_exception(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ALCHEMY_RPC_URL", "https://example.invalid")
    monkeypatch.setenv("HF_RECONCILIATION_TOLERANCE_BPS", "not-an-integer")

    app = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()
    app.sidebar.radio[1].set_value("Live Ethereum read").run()

    assert not app.exception
    assert any("Live snapshot failed" in error.value for error in app.error)
