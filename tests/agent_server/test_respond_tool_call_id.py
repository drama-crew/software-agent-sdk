from openhands.agent_server.models import ConfirmationResponseRequest


def test_request_accepts_optional_tool_call_id():
    r = ConfirmationResponseRequest(accept=True, tool_call_id="A")
    assert r.tool_call_id == "A"
    r2 = ConfirmationResponseRequest(accept=True)
    assert r2.tool_call_id is None
