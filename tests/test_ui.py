import requests

from finsight.ui import app


def test_post_retries_on_timeout(monkeypatch) -> None:
    call_count = {"count": 0}

    def fake_request(method: str, url: str, timeout: int, **kwargs):
        call_count["count"] += 1
        if call_count["count"] == 1:
            raise requests.Timeout("forced timeout")

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {"success": True, "data": {"ok": True}}

        return FakeResponse()

    monkeypatch.setattr(app._API_SESSION, "request", fake_request)

    result = app._post("/research", {"query": "test"}, timeout=1)

    assert result["data"]["ok"] is True
    assert call_count["count"] == 2
