"""Tests for per-kind ACP session map helpers (_acp_session_kind, _merge_acp_session, _read_prior_acp_session)."""

from __future__ import annotations

from openhands.sdk.agent.acp_agent import _acp_session_kind


def test_kind_normalizes_known_agents():
    assert _acp_session_kind("opencode") == "opencode"
    assert _acp_session_kind("OpenCode 1.2") == "opencode"
    assert _acp_session_kind("claude-agent-acp v0.1") == "claude"
    assert _acp_session_kind("hermes") == "hermes"
    assert _acp_session_kind("codex-acp") == "codex"
    assert _acp_session_kind("gemini-cli") == "gemini"


def test_kind_falls_back_to_slug_for_unknown():
    assert _acp_session_kind("Some New Agent!") == "some-new-agent"


def test_kind_empty_is_unknown():
    assert _acp_session_kind("") == "unknown"


# ---------------------------------------------------------------------------
# Task 1.2 — _merge_acp_session
# ---------------------------------------------------------------------------

from openhands.sdk.agent.acp_agent import _merge_acp_session  # noqa: E402


def test_merge_writes_per_kind_and_keeps_legacy():
    prev = {"unrelated": 1}
    out = _merge_acp_session(prev, kind="opencode", session_id="s1", cwd="/w")
    assert out["unrelated"] == 1
    assert out["acp_sessions"]["opencode"] == {"id": "s1", "cwd": "/w"}
    assert out["acp_session_id"] == "s1"
    assert out["acp_session_cwd"] == "/w"


def test_merge_second_kind_does_not_clobber_first():
    out = _merge_acp_session({}, kind="opencode", session_id="s1", cwd="/w")
    out = _merge_acp_session(out, kind="claude", session_id="s2", cwd="/w")
    assert out["acp_sessions"]["opencode"] == {"id": "s1", "cwd": "/w"}
    assert out["acp_sessions"]["claude"] == {"id": "s2", "cwd": "/w"}
    assert out["acp_session_id"] == "s2"


# ---------------------------------------------------------------------------
# Task 1.3 — _read_prior_acp_session
# ---------------------------------------------------------------------------

from openhands.sdk.agent.acp_agent import _read_prior_acp_session  # noqa: E402


def test_read_prefers_per_kind():
    st = {
        "acp_sessions": {"opencode": {"id": "s1", "cwd": "/w"}},
        "acp_session_id": "LEGACY",
        "acp_session_cwd": "/old",
    }
    sid, cwd = _read_prior_acp_session(st, kind="opencode")
    assert sid == "s1" and cwd == "/w"


def test_read_falls_back_to_legacy_when_kind_absent():
    st = {"acp_session_id": "LEGACY", "acp_session_cwd": "/old"}
    sid, cwd = _read_prior_acp_session(st, kind="opencode")
    assert sid == "LEGACY" and cwd == "/old"


def test_read_returns_none_when_nothing():
    sid, cwd = _read_prior_acp_session({}, kind="opencode")
    assert sid is None and cwd is None
