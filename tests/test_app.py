from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_missing_key_disables_live_chat_and_demo_can_chat(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert app.chat_input[0].disabled
    assert any("key not configured" in notice.value for notice in app.info)
    app.radio[0].set_value("Offline demo").run()
    assert not app.exception and not app.chat_input[0].disabled
    app.chat_input[0].set_value("Hello").run()
    assert not app.exception
    messages = app.session_state["messages"]
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["trace"] == []
    assert messages[-1]["latency_ms"] > 0
    next(b for b in app.button if b.label == "Clear chat").click().run()
    assert app.session_state["messages"] == []


def test_switching_provider_resets_chat_and_workspace_tokens_are_validated(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(str(APP))
    app.query_params["workspace"] = "../../other-data"
    app.run()
    assert not app.exception
    workspace = app.session_state["active_workspace"]
    assert len(workspace) == 32 and "/" not in workspace
    app.radio[0].set_value("Offline demo").run()
    app.chat_input[0].set_value("Hello").run()
    assert app.session_state["messages"]
    app.radio[0].set_value("OpenAI").run()
    assert app.session_state["messages"] == []
    assert app.chat_input[0].disabled
