import json
from typing import Any, Dict
import urllib.error
import types

from backend.automation.cicd_pipeline import CICDPipeline, PipelineConfig


class _Resp:
    def __init__(self, status: int = 200):
        self.status = status
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False


def test_post_webhook_sync_success(monkeypatch):
    pipeline = CICDPipeline(PipelineConfig(notification_webhook="http://example/webhook"))

    def fake_urlopen(req, timeout: float = 5.0):  # noqa: ARG001
        assert req.get_method() == "POST"
        assert req.headers.get("Content-Type") == "application/json"
        data = req.data
        payload = json.loads(data.decode("utf-8"))
        assert "run_id" in payload or isinstance(payload, dict)
        return _Resp(status=204)

    # Patch the module-level urlopen that CICDPipeline uses
    import backend.automation.cicd_pipeline as mod
    monkeypatch.setattr(mod._urlrequest, "urlopen", fake_urlopen)

    code = pipeline._post_webhook_sync("http://example/webhook", {"run_id": "r1"})
    assert code == 204


def test_post_webhook_sync_http_error(monkeypatch):
    pipeline = CICDPipeline(PipelineConfig(notification_webhook="http://example/webhook"))

    def fake_urlopen(*args, **kwargs):  # noqa: ARG001
        raise urllib.error.HTTPError("http://example/webhook", 503, "Service Unavailable", None, None)

    import backend.automation.cicd_pipeline as mod
    monkeypatch.setattr(mod._urlrequest, "urlopen", fake_urlopen)

    code = pipeline._post_webhook_sync("http://example/webhook", {"run_id": "r2"})
    assert code == 503


def test_post_webhook_sync_generic_error(monkeypatch):
    pipeline = CICDPipeline(PipelineConfig(notification_webhook="http://example/webhook"))

    def fake_urlopen(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("boom")

    import backend.automation.cicd_pipeline as mod
    monkeypatch.setattr(mod._urlrequest, "urlopen", fake_urlopen)

    code = pipeline._post_webhook_sync("http://example/webhook", {"run_id": "r3"})
    assert code == 599
