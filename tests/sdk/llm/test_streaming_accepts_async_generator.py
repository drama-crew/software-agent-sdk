"""Regression: LLM streaming must accept a bare (async) generator, not only
``CustomStreamWrapper``.

litellm's ``acompletion``/``completion`` return a bare ``async_generator`` /
``generator`` (rather than a ``CustomStreamWrapper``) when ``api_base`` points
at another litellm proxy -- but it yields the exact same
``ModelResponseStream`` chunks the wrapper would, so the SDK's streaming drain
loop works on it identically. The transport-call streaming branch must
therefore accept either, not assert the concrete ``CustomStreamWrapper`` type
(which crashed every managed-runtime turn that streamed through the in-cluster
litellm proxy).
"""

from litellm.types.utils import (
    Delta,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)

import openhands.sdk.llm.llm as llm_mod
from openhands.sdk.llm.llm import LLM


def _fake_chunks():
    return [
        ModelResponseStream(
            choices=[StreamingChoices(index=0, delta=Delta(content=t))]
        )
        for t in ["hel", "lo", " world"]
    ]


def _make_llm() -> LLM:
    return LLM(
        model="gpt-5.6-terra",
        api_key="x",
        base_url="http://litellm-proxy:4000",
        num_retries=1,
    )


async def test_atransport_call_accepts_bare_async_generator(monkeypatch):
    async def _agen():
        for c in _fake_chunks():
            yield c

    async def fake_acompletion(**kwargs):
        # A litellm-proxy route returns a bare async_generator, NOT a
        # CustomStreamWrapper.
        return _agen()

    monkeypatch.setattr(llm_mod, "litellm_acompletion", fake_acompletion)

    llm = _make_llm()
    tokens: list = []
    res = await llm._atransport_call(
        messages=[{"role": "user", "content": "hi"}],
        enable_streaming=True,
        on_token=lambda c: tokens.append(c),
    )
    assert isinstance(res, ModelResponse)
    assert len(tokens) == 3  # on_token fired per chunk
    assembled = "".join((c.choices[0].delta.content or "") for c in tokens)
    assert "hello world" == assembled


def test_transport_call_accepts_bare_sync_generator(monkeypatch):
    def _gen():
        for c in _fake_chunks():
            yield c

    def fake_completion(**kwargs):
        return _gen()

    monkeypatch.setattr(llm_mod, "litellm_completion", fake_completion)

    llm = _make_llm()
    tokens: list = []
    res = llm._transport_call(
        messages=[{"role": "user", "content": "hi"}],
        enable_streaming=True,
        on_token=lambda c: tokens.append(c),
    )
    assert isinstance(res, ModelResponse)
    assert len(tokens) == 3
