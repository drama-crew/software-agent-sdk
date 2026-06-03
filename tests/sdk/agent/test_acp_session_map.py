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
