import os
from typing import Optional
from fastapi import FastAPI
from fastapi.testclient import TestClient


def make_client(admin_key: Optional[str]) -> TestClient:
    """Build a minimal FastAPI app with the CI/CD router and desired admin key.

    Resets the centralized settings singleton so the router reads the updated env.
    """
    if admin_key is None:
        os.environ.pop("ADMIN_API_KEY", None)
    else:
        os.environ["ADMIN_API_KEY"] = admin_key

    # Force settings reload
    import backend.core.config as cfg
    cfg._SETTINGS = None  # type: ignore[attr-defined]

    from backend.routes.cicd_management import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_unauthorized_without_header():
    client = make_client("test_key")
    r = client.get("/api/cicd/status")
    assert r.status_code == 401


def test_unauthorized_wrong_key_header():
    client = make_client("test_key")
    r = client.get("/api/cicd/status", headers={"X-Admin-API-Key": "wrong"})
    assert r.status_code == 401


def test_authorized_with_correct_header():
    client = make_client("test_key")
    r = client.get("/api/cicd/status", headers={"X-Admin-API-Key": "test_key"})
    assert r.status_code == 200
    body = r.json()
    assert body.get("success") is True


def test_authorized_with_bearer_token():
    client = make_client("test_key")
    r = client.get("/api/cicd/status", headers={"Authorization": "Bearer test_key"})
    assert r.status_code == 200
    body = r.json()
    assert body.get("success") is True


def test_reject_when_admin_key_not_configured():
    client = make_client(None)
    r = client.get("/api/cicd/status")
    # 403 when no admin key configured
    assert r.status_code == 403
