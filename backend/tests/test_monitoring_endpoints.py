import os
import asyncio
from typing import Callable, Awaitable, Any
import sys
import types
import pathlib
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def pm(monkeypatch):
    """Fresh PerformanceMonitor instance with test thresholds.
    Ensures the singleton is not reused across tests.
    """
    # Configure environment thresholds for this test instance
    os.environ["ARIA_THRESH_CPU_WARN"] = "77"
    os.environ["ARIA_THRESH_MEM_AVAIL_CRIT"] = "12"
    os.environ["ARIA_THRESH_LATENCY_WARN_MS"] = "1234"

    from backend.core import performance_monitor as pm_mod

    # Reset singleton and construct a fresh monitor to pick up env
    pm_mod._performance_monitor = None
    monitor = pm_mod.PerformanceMonitor()
    try:
        yield monitor
    finally:
        # Cleanup: drop any active connections
        monitor.active_connections.clear()


@pytest.fixture()
def client(pm, monkeypatch) -> TestClient:
    """FastAPI TestClient mounting only the monitoring router.
    We patch get_performance_monitor to return our test instance.
    Optionally, we stub auto_trader to avoid heavy side effects unless
    RUN_AUTOTRADER_INTEGRATION=1 is set in the environment.
    """
    use_integration = os.getenv("RUN_AUTOTRADER_INTEGRATION", "0") == "1"

    # Stub heavy auto_trader import before loading the router to avoid side effects
    if not use_integration:
        fake_at = types.ModuleType("backend.services.auto_trader")
        class _DummyAT:
            def get_status(self):
                return {"running": False}
        fake_at.auto_trader = _DummyAT()
        monkeypatch.setitem(sys.modules, "backend.services.auto_trader", fake_at)

    from backend.routes import monitoring

    # Patch router helper to return our prepared monitor instance
    monkeypatch.setattr(monitoring, "get_performance_monitor", lambda: pm)

    app = FastAPI()
    app.include_router(monitoring.router)
    return TestClient(app)


def _run(coro_factory: Callable[[], Awaitable[Any]]) -> None:
    """Run a coroutine to completion on a dedicated event loop.
    Keeps the router isolated from uvicorn/Starlette loops under TestClient.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(coro_factory())
    finally:
        loop.close()


def test_thresholds_endpoint_returns_env_values(client: TestClient, pm) -> None:
    # Validate thresholds are those from env on our fresh monitor
    r = client.get("/monitoring/performance/thresholds")
    assert r.status_code == 200
    data = r.json()
    assert pytest.approx(data["cpu_warn_percent"], rel=1e-6) == 77.0
    assert pytest.approx(data["mem_available_percent_crit"], rel=1e-6) == 12.0
    assert pytest.approx(data["latency_warn_ms"], rel=1e-6) == 1234.0


def test_per_symbol_and_model_metrics_endpoints(client: TestClient, pm) -> None:
    # Populate per-model and per-symbol metrics via the monitor's async API
    def _populate():
        return pm._update_metrics(  # noqa: SLF001 test-only internal use
            model_name="xgb",
            latency_ms=512.0,
            cpu_metrics={"cpu_percent_avg": 35.0, "memory_mb_avg": 180.0},
            symbol="EURUSD",
        )

    _run(_populate)

    # GET /performance/metrics -> should include system, models, symbols, thresholds
    r = client.get("/monitoring/performance/metrics")
    assert r.status_code == 200
    j = r.json()
    assert "system" in j and "models" in j and "symbols" in j and "thresholds" in j
    assert "EURUSD" in j["symbols"]
    assert "xgb" in j["models"]

    # GET /performance/symbols -> map of symbol -> model -> metrics
    r = client.get("/monitoring/performance/symbols")
    assert r.status_code == 200
    symbols_payload = r.json()
    assert "EURUSD" in symbols_payload
    assert "xgb" in symbols_payload["EURUSD"]
    assert symbols_payload["EURUSD"]["xgb"]["model"] == "xgb"

    # GET /performance/symbols/{symbol}
    r = client.get("/monitoring/performance/symbols/EURUSD")
    assert r.status_code == 200
    sym_payload = r.json()
    assert "xgb" in sym_payload
    assert sym_payload["xgb"]["latency_ms"]["avg"] >= 0

    # GET /performance/models/{model_name}
    r = client.get("/monitoring/performance/models/xgb")
    assert r.status_code == 200
    model_metrics = r.json()
    assert model_metrics["model"] == "xgb"
    assert model_metrics["calls"] >= 1


def test_auto_trader_status_endpoint(client: TestClient, monkeypatch) -> None:
    # Enabled should reflect env var
    monkeypatch.setenv("AUTO_TRADE_ENABLED", "1")
    r = client.get("/monitoring/auto-trader/status")
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    assert data.get("enabled") is True

    monkeypatch.setenv("AUTO_TRADE_ENABLED", "0")
    r = client.get("/monitoring/auto-trader/status")
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    assert data.get("enabled") is False

    # Error path: make auto_trader.get_status raise
    if os.getenv("RUN_AUTOTRADER_INTEGRATION", "0") == "1":
        pytest.skip("Skip stubbed error-path under integration mode")
    from backend.routes import monitoring as mon
    class _Err:
        def get_status(self):
            raise RuntimeError("boom")
    monkeypatch.setattr(mon, "auto_trader", _Err())
    r = client.get("/monitoring/auto-trader/status")
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is False
    assert "error" in data


def test_models_status_endpoint_basic(client: TestClient) -> None:
    # Ensure the lightweight models status endpoint returns structure without loading models
    r = client.get("/monitoring/models/status")
    assert r.status_code == 200
    payload = r.json()
    assert payload.get("ok") is True
    models = payload.get("models")
    assert isinstance(models, dict)
    # Check for a couple of known keys and basic fields
    for key in ["cnn", "xgb"]:
        assert key in models
        item = models[key]
        assert set(["name", "model_path", "exists", "modules_available"]).issubset(item.keys())


def test_websocket_performance_stream(client: TestClient) -> None:
    # Connect and receive at least one system_metrics message, then ping/pong
    with client.websocket_connect("/monitoring/ws/performance") as ws:
        msg = ws.receive_json()
        assert isinstance(msg, dict)
        assert msg.get("type") == "system_metrics"

        ws.send_text("ping")
        pong = ws.receive_json()
        assert pong.get("type") == "pong" or pong.get("type") == None  # handler sends {type:pong}


@pytest.mark.skipif(os.getenv("RUN_AUTOTRADER_INTEGRATION", "0") != "1", reason="Integration test requires RUN_AUTOTRADER_INTEGRATION=1 and MT5 env")
def test_auto_trader_status_endpoint_integration(client: TestClient, monkeypatch) -> None:
    """Integration variant using real backend.services.auto_trader import.
    Skipped unless RUN_AUTOTRADER_INTEGRATION=1 is set and MT5 environment is configured.
    """
    monkeypatch.setenv("AUTO_TRADE_ENABLED", "1")
    r = client.get("/monitoring/auto-trader/status")
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") in (True, False)  # allow ok=False if runtime constraints
    assert data.get("enabled") is True


def test_metrics_prom_latency_and_decisions_with_temp_logs(client: TestClient, monkeypatch, tmp_path) -> None:
    # Point monitoring DATA_ROOT to a temporary directory
    from backend.routes import monitoring as mon
    tmp_data_root = pathlib.Path(tmp_path)
    monkeypatch.setattr(mon, "DATA_ROOT", tmp_data_root)

    # Create live_logs and a decisions_*.jsonl file
    log_dir = tmp_data_root / "live_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "decisions_20250101.jsonl"
    lines = [
        {"lat_ms": 25.0, "action": "LONG", "symbol": "EURUSD"},
        {"lat_ms": 55.0, "action": "SHORT", "symbol": "GBPUSD"},
        "not-json",  # ensure parser skips bad lines
    ]
    with log_file.open("w", encoding="utf-8") as f:
        for item in lines:
            if isinstance(item, dict):
                f.write(__import__("json").dumps(item) + "\n")
            else:
                f.write(str(item) + "\n")

    # Test Prometheus metrics endpoint
    r = client.get("/monitoring/metrics/prom")
    assert r.status_code == 200
    text = r.text
    assert "aria_info" in text
    assert 'aria_latency_ms{quantile="0.5"}' in text
    assert 'aria_actions_total{action="LONG"}' in text
    assert 'aria_actions_total{action="SHORT"}' in text

    # Deterministic checks for quantiles and counters
    assert 'aria_latency_ms{quantile="0.5"} 25.0' in text
    assert 'aria_latency_ms{quantile="0.95"} 55.0' in text
    assert 'aria_latency_ms{quantile="0.99"} 55.0' in text
    assert 'aria_latency_ms_sum 80.0' in text
    assert 'aria_latency_ms_count 2' in text
    assert 'aria_actions_total{action="LONG"} 1' in text
    assert 'aria_actions_total{action="SHORT"} 1' in text
    assert 'aria_actions_total{action="FLAT"} 0' in text

    # Test latency summary
    r = client.get("/monitoring/latency")
    assert r.status_code == 200
    payload = r.json()
    assert payload["count"] == 2
    assert payload["p50"] is not None and payload["p95"] is not None and payload["p99"] is not None

    # Test decisions/last all
    r = client.get("/monitoring/decisions/last")
    assert r.status_code == 200
    all_items = r.json()
    assert all_items["count"] == 2
    assert len(all_items["items"]) == 2

    # Test decisions/last filtered by symbol
    r = client.get("/monitoring/decisions/last", params={"symbol": "EURUSD"})
    assert r.status_code == 200
    sym_items = r.json()
    assert sym_items["count"] >= 1
    assert sym_items["items"] and sym_items["items"][-1]["symbol"] == "EURUSD"


def test_metrics_prom_no_logs_emits_info_only(client: TestClient, monkeypatch, tmp_path) -> None:
    from backend.routes import monitoring as mon
    tmp_data_root = pathlib.Path(tmp_path)
    # No live_logs created -> metrics_prom should emit only aria_info
    monkeypatch.setattr(mon, "DATA_ROOT", tmp_data_root)

    r = client.get("/monitoring/metrics/prom")
    assert r.status_code == 200
    text = r.text
    assert "aria_info" in text
    assert "aria_latency_ms" not in text
    assert "aria_actions_total" not in text


def test_log_rotation_prefers_latest_file(client: TestClient, monkeypatch, tmp_path) -> None:
    from backend.routes import monitoring as mon
    tmp_data_root = pathlib.Path(tmp_path)
    monkeypatch.setattr(mon, "DATA_ROOT", tmp_data_root)

    log_dir = tmp_data_root / "live_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    f_old = log_dir / "decisions_20240101.jsonl"
    f_new = log_dir / "decisions_20250102.jsonl"

    with f_old.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"lat_ms": 10.0, "action": "LONG", "symbol": "EURUSD"}) + "\n")
        f.write("badline\n")

    with f_new.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"lat_ms": 90.0, "action": "SHORT", "symbol": "USDJPY"}) + "\n")
        f.write(json.dumps({"lat_ms": 100.0, "action": "LONG", "symbol": "EURUSD"}) + "\n")

    # Prometheus metrics should reflect only the latest file
    r = client.get("/monitoring/metrics/prom")
    assert r.status_code == 200
    text = r.text
    assert 'aria_latency_ms{quantile="0.5"} 90.0' in text
    assert 'aria_latency_ms{quantile="0.95"} 100.0' in text
    assert 'aria_latency_ms{quantile="0.99"} 100.0' in text
    assert 'aria_latency_ms_sum 190.0' in text
    assert 'aria_latency_ms_count 2' in text
    assert 'aria_actions_total{action="LONG"} 1' in text
    assert 'aria_actions_total{action="SHORT"} 1' in text
    assert 'aria_actions_total{action="FLAT"} 0' in text

    # Latency summary should use latest file
    r = client.get("/monitoring/latency")
    assert r.status_code == 200
    payload = r.json()
    assert payload["count"] == 2
    assert payload["p50"] == 90.0
    assert payload["p95"] == 100.0
    assert payload["p99"] == 100.0

    # Decisions last should also use the latest file
    r = client.get("/monitoring/decisions/last")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2


def test_perf_monitor_latency_threshold_alerts_broadcasts(client: TestClient, pm, monkeypatch) -> None:
    # Avoid CPU/memory alerts; focus on latency
    pm.thresholds["cpu_warn_percent"] = 1000.0
    pm.thresholds["mem_available_percent_crit"] = 0.0
    pm.thresholds["latency_warn_ms"] = 50.0

    messages = []

    async def _fake_broadcast(msg):
        messages.append(msg)

    # Patch the monitor's broadcaster
    monkeypatch.setattr(pm, "_broadcast_metrics", _fake_broadcast)

    # Populate metrics above threshold, including per-symbol
    def _populate_high():
        return pm._update_metrics(
            model_name="xgb",
            latency_ms=75.0,
            cpu_metrics={"cpu_percent_avg": 0.0, "memory_mb_avg": 0.0},
            symbol="EURUSD",
        )

    _run(_populate_high)

    # Trigger threshold checks
    _run(lambda: pm.check_thresholds())

    alerts = [m for m in messages if isinstance(m, dict) and m.get("type") == "alert"]
    assert any("High latency in xgb" in m.get("message", "") for m in alerts)
    assert any("High latency EURUSD/xgb" in m.get("message", "") for m in alerts)

    # Now raise threshold so no alerts fire
    messages.clear()
    pm.thresholds["latency_warn_ms"] = 1e9
    _run(lambda: pm.check_thresholds())
    alerts = [m for m in messages if isinstance(m, dict) and m.get("type") == "alert"]
    assert not alerts


def test_prometheus_edge_quantiles_extreme_latencies(client: TestClient, monkeypatch, tmp_path) -> None:
    """Test Prometheus metrics with extreme latency values for edge quantiles (0.01, 0.999)."""
    from backend.routes import monitoring as mon
    tmp_data_root = pathlib.Path(tmp_path)
    monkeypatch.setattr(mon, "DATA_ROOT", tmp_data_root)

    log_dir = tmp_data_root / "live_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "decisions_20250103.jsonl"
    
    # Generate 1000 latency values with extreme outliers
    latencies = []
    # 970 normal values (10-99ms)
    for i in range(970):
        latencies.append({"lat_ms": 10.0 + (i % 90), "action": "LONG" if i % 3 == 0 else "SHORT", "symbol": "EURUSD"})
    # 15 extremely low outliers (0.1-2.9ms)
    for i in range(15):
        latencies.append({"lat_ms": 0.1 + i * 0.2, "action": "FLAT", "symbol": "GBPUSD"})
    # 15 extremely high outliers (5000-19000ms)
    for i in range(15):
        latencies.append({"lat_ms": 5000.0 + i * 1000, "action": "LONG", "symbol": "USDJPY"})
    
    with log_file.open("w", encoding="utf-8") as f:
        for item in latencies:
            f.write(json.dumps(item) + "\n")
    
    # Test Prometheus metrics with edge quantiles
    r = client.get("/monitoring/metrics/prom")
    assert r.status_code == 200
    text = r.text
    
    # Standard quantiles should be in normal range; edge quantiles should reflect outliers
    lines = text.split('\n')
    def _qval(q: str) -> float:
        for line in lines:
            if f'aria_latency_ms{{quantile="{q}"}}' in line:
                return float(line.split()[-1])
        raise AssertionError(f"Quantile {q} not found in Prometheus output")

    assert 10.0 <= _qval("0.5") <= 100.0
    # 1% quantile should land among low outliers (<= ~3ms)
    assert _qval("0.01") <= 3.0
    # 99.9% quantile should land among high outliers (>= 5000ms)
    assert _qval("0.999") >= 5000.0
    
    # Verify we have 1000 samples
    assert 'aria_latency_ms_count 1000' in text
    
    # Test latency endpoint with extreme values
    r = client.get("/monitoring/latency")
    assert r.status_code == 200
    payload = r.json()
    assert payload["count"] == 1000
    # p50 should be in normal range
    assert 10.0 <= payload["p50"] <= 100.0
    # With 15 high outliers, p99 should be among outliers
    assert payload["p99"] >= 5000.0


def test_websocket_alert_escalation_realtime(client: TestClient, pm, monkeypatch) -> None:
    """Test WebSocket real-time streaming and alert escalation when thresholds are crossed."""
    # Set aggressive thresholds for immediate alerts
    pm.thresholds["cpu_warn_percent"] = 50.0
    pm.thresholds["latency_warn_ms"] = 100.0
    pm.thresholds["mem_available_percent_crit"] = 50.0
    
    received_messages = []
    
    with client.websocket_connect("/monitoring/ws/performance") as ws:
        # Should receive initial system_metrics message
        msg = ws.receive_json()
        assert msg.get("type") == "system_metrics"
        received_messages.append(msg)
        
        # Simulate high latency to trigger alert
        def _simulate_high_latency():
            return pm._update_metrics(
                model_name="lstm",
                latency_ms=200.0,  # Above 100ms threshold
                cpu_metrics={"cpu_percent_avg": 60.0, "memory_mb_avg": 512.0},
                symbol="EURUSD",
            )
        
        _run(_simulate_high_latency)
        
        # Force threshold check to broadcast alerts
        _run(lambda: pm.check_thresholds())
        
        # Receive a few messages that should include model metrics and alerts
        for _ in range(3):
            try:
                received_messages.append(ws.receive_json())
            except Exception:
                break
        
        # Verify we received model metrics and alerts
        model_metrics = [m for m in received_messages if m.get("type") == "model_metrics"]
        alerts = [m for m in received_messages if m.get("type") == "alert"]
        
        assert len(model_metrics) > 0, "Should receive model metrics"
        assert any(m.get("model") == "lstm" for m in model_metrics)
        
        # Should have latency alerts
        assert len(alerts) > 0, "Should receive alert messages"
        alert_messages = [a.get("message", "") for a in alerts]
        assert any("High latency" in msg and "lstm" in msg for msg in alert_messages)


def test_websocket_multiple_clients_broadcast(client: TestClient, pm) -> None:
    """Test that multiple WebSocket clients all receive broadcast messages."""
    clients_data = []
    
    # Connect multiple clients
    with client.websocket_connect("/monitoring/ws/performance") as ws1:
        with client.websocket_connect("/monitoring/ws/performance") as ws2:
            # Both should receive initial system metrics
            msg1 = ws1.receive_json()
            msg2 = ws2.receive_json()
            assert msg1.get("type") == "system_metrics"
            assert msg2.get("type") == "system_metrics"
            
            # Trigger a model update
            def _update():
                return pm._update_metrics(
                    model_name="xgb",
                    latency_ms=50.0,
                    cpu_metrics={"cpu_percent_avg": 25.0, "memory_mb_avg": 128.0},
                    symbol="GBPUSD",
                )
            
            _run(_update)
            
            # Both clients should receive the update (read a few messages)
            msgs1 = [ws1.receive_json() for _ in range(2)]
            msgs2 = [ws2.receive_json() for _ in range(2)]
            
            # Both should have received model metrics
            assert any(m.get("type") == "model_metrics" for m in msgs1)
            assert any(m.get("type") == "model_metrics" for m in msgs2)


def test_alerts_with_anomalies_monkeypatched(client: TestClient, monkeypatch) -> None:
    from backend.routes import monitoring as mon

    def _fake_metrics():
        return {
            "cpu_usage": 85.1,
            "memory_usage": 90.2,
            "network_io": {"bytes_sent": 0, "bytes_recv": 0},
            "uptime": 10.0,
            "trading_metrics": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0.0,
                "daily_pnl": 0.0,
                "max_drawdown": 0.0,
            },
            "system_health": {
                "mt5_connected": False,
                "orchestrator_running": False,
                "fusion_cores_active": 0,
                "last_heartbeat": 0.0,
            },
        }

    monkeypatch.setattr(mon, "get_system_metrics", _fake_metrics)

    r = client.get("/monitoring/alerts")
    assert r.status_code == 200
    alerts = r.json()
    messages = {a.get("message") for a in alerts}
    # Expect 4 alerts with specific messages
    assert any(m.startswith("High CPU usage: 85.1%") for m in messages)
    assert any(m.startswith("High memory usage: 90.2%") for m in messages)
    assert "MT5 connection lost" in messages
    assert "Phase 3 Orchestrator not running" in messages


def test_status_and_gating_with_temp_roots(client: TestClient, monkeypatch, tmp_path) -> None:
    from backend.routes import monitoring as mon

    # Create temporary DATA_ROOT with calibration manifests
    tmp_data_root = pathlib.Path(tmp_path)
    calib = tmp_data_root / "calibration" / "current"
    (calib / "EURUSD").mkdir(parents=True, exist_ok=True)
    (calib / "GBPUSD").mkdir(parents=True, exist_ok=True)
    (calib / "EURUSD" / "fusion_lr.json").write_text(
        json.dumps({
            "type": "logreg",
            "version_hash": "abc123",
            "features_order": ["f1", "f2", "f3"],
        }),
        encoding="utf-8",
    )
    (calib / "GBPUSD" / "fusion_lr.json").write_text(
        json.dumps({
            "type": "logreg",
            "version_hash": "def456",
            "features_order": ["v1", "v2"],
        }),
        encoding="utf-8",
    )

    # Create temporary gating JSON
    gating_path = tmp_data_root / "gating.test.json"
    gating_payload = {
        "version": "1.0-test",
        "default_thresholds": {"foo": 1, "bar": 2},
    }
    gating_path.write_text(json.dumps(gating_payload), encoding="utf-8")

    monkeypatch.setattr(mon, "DATA_ROOT", tmp_data_root)
    monkeypatch.setattr(mon, "GATING_PATH", gating_path)

    # /monitoring/status
    r = client.get("/monitoring/status")
    assert r.status_code == 200
    payload = r.json()
    assert payload["data_root"] == str(tmp_data_root)
    assert payload["gating_version"] == gating_payload["version"]
    assert payload["thresholds"] == gating_payload["default_thresholds"]
    assert sorted(payload["symbols"]) == ["EURUSD", "GBPUSD"]
    manifest = payload["calibration_manifest"]
    assert "EURUSD" in manifest and "GBPUSD" in manifest
    assert manifest["EURUSD"]["features_order"] == ["f1", "f2", "f3"]

    # /monitoring/gating
    r = client.get("/monitoring/gating")
    assert r.status_code == 200
    assert r.json() == gating_payload
