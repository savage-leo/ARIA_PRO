#!/usr/bin/env python3
"""
DAN Audit Engine - production preflight scanner for ARIA PRO.
Usage:
  python backend\tools\dan_audit_engine.py --report <out.json> [--auto-fix]

What it does:
- Runs formatting/type checks: black --check, mypy, flake8 (if available)
- Runs pytest (backend)
- Scans repo for simulation/fake/demo fallback patterns
- Checks for unpinned critical dependencies in backend/requirements.txt
- Attempts MT5 connection (if MetaTrader5 installed). If no MT5 -> logs alert and exits code 2.
- If MT5 is OK, runs model smoke tests (load ONNX / PT / TF models and record load status)
- Outputs JSON report to --report path
"""
import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root

SIM_PATTERNS = [
    r"\bsimulat(e|ion)\b",
    r"\bfake\b",
    r"\bdemo\b",
    r"\bmock\b",
    r"\bfallback\b",
    r"\buse_simulator\b",
    r"\bis_simulation\b",
    r"\bget_mock_data\b",
    r"\bsimulate_market\b",
    r"\bsynthetic_data\b",
]

CRITICAL_UNPINNED = ["fastapi", "pybind11", "tensorflow"]


def run_cmd(cmd: str, cwd: str | None = None, timeout: int = 300) -> tuple[int, str]:
    try:
        res = subprocess.run(
            cmd, cwd=cwd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return res.returncode, (res.stdout or "") + (
            "\n" + res.stderr if res.stderr else ""
        )
    except Exception as e:  # pragma: no cover
        return 255, str(e)


def search_simulation_patterns() -> list[dict]:
    print("Searching for simulation/mocks/fallbacks...")
    hits: list[dict] = []
    for p in ROOT.rglob("*.*"):
        parts = set(p.parts)
        if any(
            part.startswith(".git")
            or part in ("node_modules", "venv", ".venv", "__pycache__")
            for part in parts
        ):
            continue
        if p.suffix.lower() not in (
            ".py",
            ".ts",
            ".tsx",
            ".js",
            ".json",
            ".yaml",
            ".yml",
        ):
            continue
        try:
            txt = p.read_text(encoding="utf8", errors="ignore")
        except Exception:
            continue
        for pat in SIM_PATTERNS:
            for m in re.finditer(pat, txt, re.IGNORECASE):
                start = max(0, m.start() - 120)
                end = min(len(txt), m.end() + 120)
                ctx = txt[start:end].replace("\n", " ")
                hits.append(
                    {
                        "file": str(p.relative_to(ROOT)),
                        "pattern": pat,
                        "match": m.group(0),
                        "context": ctx,
                    }
                )
    return hits


def check_requirements_unpinned() -> list[str]:
    req = ROOT / "backend" / "requirements.txt"
    unpinned: list[str] = []
    if req.exists():
        text = req.read_text(encoding="utf8", errors="ignore")
        for crit in CRITICAL_UNPINNED:
            if re.search(rf"(?mi)^{crit}\\b(?!.*==)", text):
                unpinned.append(crit)
    return unpinned


def run_linters_and_tests(report: dict) -> None:
    report["linters"] = {}

    # black --check
    code, _ = run_cmd("python -m black --version")
    if code == 0:
        code, out = run_cmd("python -m black --check .")
        report["linters"]["black"] = {"rc": code, "output": out}
    else:
        report["linters"]["black"] = {"rc": -1, "output": "black not installed"}

    # mypy
    code, _ = run_cmd("python -m mypy --version")
    if code == 0:
        code, out = run_cmd("python -m mypy ./backend --ignore-missing-imports")
        report["linters"]["mypy"] = {"rc": code, "output": out}
    else:
        report["linters"]["mypy"] = {"rc": -1, "output": "mypy not installed"}

    # flake8
    code, _ = run_cmd("python -m flake8 --version")
    if code == 0:
        code, out = run_cmd("python -m flake8 backend --max-line-length=120")
        report["linters"]["flake8"] = {"rc": code, "output": out}
    else:
        report["linters"]["flake8"] = {"rc": -1, "output": "flake8 not installed"}

    # pytest (backend only)
    if (ROOT / "backend").exists():
        code, out = run_cmd(
            "python -m pytest backend -q --maxfail=1 --disable-warnings --capture=no"
        )
        report["tests"] = {"rc": code, "output": out}
    else:
        report["tests"] = {"rc": -1, "output": "backend folder not found"}


def mt5_connect_check() -> tuple[bool, str]:
    try:
        import MetaTrader5 as mt5  # type: ignore
    except Exception as e:  # pragma: no cover
        return False, f"MetaTrader5 module not installed: {e}"
    try:
        ok = mt5.initialize()
        if not ok:
            return False, f"mt5.initialize() failed: {mt5.last_error()}"
        mt5.shutdown()
        return True, "MT5 module available and initialized (no credentials)."
    except Exception as e:  # pragma: no cover
        return False, f"MT5 initialize failed: {e}"


def model_smoke_tests(report: dict) -> None:
    model_report: dict = {}
    models_folder = ROOT / "data" / "models"
    model_files: list[Path] = []
    if models_folder.exists():
        model_files += list(models_folder.rglob("*.onnx"))
        model_files += list(models_folder.rglob("*.pt"))
        model_files += list(models_folder.rglob("*.pb"))
    model_report["found_models"] = [str(p.relative_to(ROOT)) for p in model_files]
    model_report["smoke_results"] = []
    for m in model_files:
        try:
            if m.suffix == ".onnx":
                import onnxruntime as ort  # type: ignore

                _ = ort.InferenceSession(str(m))
                model_report["smoke_results"].append(
                    {"model": str(m.relative_to(ROOT)), "status": "loaded_onnx"}
                )
            elif m.suffix == ".pt":
                import torch  # type: ignore

                _ = torch.load(str(m), map_location="cpu")
                model_report["smoke_results"].append(
                    {"model": str(m.relative_to(ROOT)), "status": "loaded_pt"}
                )
            else:
                model_report["smoke_results"].append(
                    {"model": str(m.relative_to(ROOT)), "status": "unknown_format"}
                )
        except Exception as e:
            model_report["smoke_results"].append(
                {
                    "model": str(m.relative_to(ROOT)),
                    "status": "error",
                    "error": str(e),
                }
            )
    report["models"] = model_report


def attempt_auto_fix_simulation(hits: list[dict]) -> list[dict]:
    """
    Limited auto-fix: for simple patterns where code uses a flag like `if use_simulation:`
    we will insert a logging and raise to prevent fallback. This is conservative.
    """
    patched: list[dict] = []
    for h in hits:
        p = ROOT / h["file"]
        try:
            txt = p.read_text(encoding="utf8", errors="ignore")
        except Exception:
            continue
        newtxt = re.sub(
            r"if\s+use_simulation\s*:\s*",
            'if use_simulation: raise RuntimeError("SIMULATION-FALLBACK-DISABLED"); ',
            txt,
        )
        if newtxt != txt:
            backup = str(p) + ".bak"
            p.write_text(newtxt, encoding="utf8")
            Path(backup).write_text(txt, encoding="utf8")
            patched.append({"file": str(p.relative_to(ROOT)), "backup": backup})
    return patched


def check_frontend_tests() -> dict:
    fe = ROOT / "frontend"
    if not fe.exists():
        return {"exists": False}
    pkg = fe / "package.json"
    if not pkg.exists():
        return {"exists": True, "package_json": False}
    txt = pkg.read_text(encoding="utf8", errors="ignore")
    has_test = ('"test"' in txt) or ("vitest" in txt) or ("jest" in txt)
    return {"exists": True, "package_json": True, "has_test_script": bool(has_test)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--auto-fix", action="store_true", default=False)
    args = ap.parse_args()

    report: dict = {
        "repo": str(ROOT),
        "ok": False,
        "errors": [],
        "simulation_hits": [],
        "linters": {},
        "tests": {},
        "models": {},
        "mt5": {},
        "frontend": {},
    }
    try:
        # 1) scan for simulation patterns
        sim_hits = search_simulation_patterns()
        report["simulation_hits"] = sim_hits
        report["simulation_count"] = len(sim_hits)

        # 2) check unpinned critical deps
        unpinned = check_requirements_unpinned()
        report["unpin_problems"] = unpinned

        # 3) run linters and tests
        run_linters_and_tests(report)

        # 4) frontend test presence check
        report["frontend"] = check_frontend_tests()

        # 5) MT5 connect check
        mt5_ok, mt5_msg = mt5_connect_check()
        report["mt5"] = {"ok": mt5_ok, "msg": mt5_msg}
        if not mt5_ok:
            with open(args.report, "w", encoding="utf8") as f:
                json.dump(report, f, indent=2)
            print("MT5 not available. Aborting model smoke tests. Report written.")
            sys.exit(2)

        # 6) model smoke tests (only if MT5 OK)
        model_smoke_tests(report)

        # 7) auto-fix simulation if requested
        if args.auto_fix and report.get("simulation_count", 0) > 0:
            patched = attempt_auto_fix_simulation(report["simulation_hits"])  # type: ignore[arg-type]
            report["auto_fixed"] = patched

        report["ok"] = True
    except Exception as e:  # pragma: no cover
        traceback.print_exc()
        report["errors"].append(str(e))
        report["ok"] = False
    with open(args.report, "w", encoding="utf8") as f:
        json.dump(report, f, indent=2)
    print("Wrote report to", args.report)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()

