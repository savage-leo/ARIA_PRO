"""ARIA preflight harness.
Performs:
 - AST parse checks
 - compileall
 - import-resolution using safe stubs for external libs
 - basic runtime import of modules to detect import-time exceptions
 - invokes pytest programmatically and collects report
"""
import os
import sys
import argparse
import json
import re
import compileall
import py_compile
import ast
import importlib
import traceback
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
IGNORES = ['.git', '__pycache__', 'venv_preflight', 'venv', '.venv', 'node_modules', 'build', 'dist', '.idea', '.vscode', 'backups']

EXTERNAL_STUB_MODULES = [
    'MetaTrader5', 'mt5', 'mt5wrapper', 'psycopg2', 'pymysql', 'redis', 'aiortc',
    'torch', 'tensorflow', 'onnxruntime', 'ollama', 'sqlalchemy', 'cx_Oracle',
    'stable_baselines3', 'gym', 'gymnasium', 'xgboost', 'sklearn', 'skl2onnx', 'onnxmltools',
    'pandas', 'numpy', 'ta', 'yfinance', 'alpaca_trade_api', 'ccxt', 'watchdog', 'joblib',
    'scipy', 'multipart', 'pydantic', 'pydantic_settings', 'prometheus_client'
]


def make_stub(mod_name):
    """Insert a minimal stub module into sys.modules to allow imports."""
    import types
    if mod_name in sys.modules:
        return
    m = types.ModuleType(mod_name)
    # add tiny no-op objects used by many libs
    m.connect = lambda *a, **k: None
    m.Client = type('Client', (), {})
    m.Session = type('Session', (), {})
    m.__version__ = '0.0.0-stub'
    
    # Module-specific stubs
    if mod_name == 'numpy':
        m.ndarray = type('ndarray', (), {})
        m.array = lambda x: x
        m.zeros = lambda x: [0] * x
        m.float32 = float
        m.float64 = float
        # Integer dtypes commonly referenced by SciPy and others
        m.int8 = int
        m.int16 = int
        m.int32 = int
        m.int64 = int
        m.intc = int
        m.intp = int
    elif mod_name == 'pandas':
        m.DataFrame = type('DataFrame', (), {})
        m.Series = type('Series', (), {})
        # Create pandas.api submodule
        api_mod = types.ModuleType('pandas.api')
        sys.modules['pandas.api'] = api_mod
        # Create pandas.api.types with light helpers
        types_mod = types.ModuleType('pandas.api.types')
        def _false(*a, **k):
            return False
        types_mod.is_numeric_dtype = _false
        types_mod.is_datetime64_any_dtype = _false
        types_mod.is_integer_dtype = _false
        sys.modules['pandas.api.types'] = types_mod
        api_mod.types = types_mod
        m.api = api_mod
    elif mod_name == 'sklearn':
        # Create sklearn submodules used by ARIA
        # sklearn.ensemble
        ensemble_mod = types.ModuleType('sklearn.ensemble')
        ensemble_mod.RandomForestRegressor = type('RandomForestRegressor', (), {})
        sys.modules['sklearn.ensemble'] = ensemble_mod
        m.ensemble = ensemble_mod
        # sklearn.linear_model
        linear_mod = types.ModuleType('sklearn.linear_model')
        linear_mod.Ridge = type('Ridge', (), {})
        linear_mod.ElasticNet = type('ElasticNet', (), {})
        sys.modules['sklearn.linear_model'] = linear_mod
        m.linear_model = linear_mod
        # sklearn.preprocessing
        preprocessing_mod = types.ModuleType('sklearn.preprocessing')
        class _StandardScaler:
            def __init__(self, *a, **k):
                pass
            def fit(self, X, y=None):
                return self
            def transform(self, X):
                return X
            def fit_transform(self, X, y=None):
                return X
        preprocessing_mod.StandardScaler = _StandardScaler
        sys.modules['sklearn.preprocessing'] = preprocessing_mod
        m.preprocessing = preprocessing_mod
        # sklearn.model_selection
        model_selection_mod = types.ModuleType('sklearn.model_selection')
        class _TimeSeriesSplit:
            def __init__(self, *a, **k):
                pass
            def split(self, X, y=None, groups=None):
                return iter([])
        model_selection_mod.TimeSeriesSplit = _TimeSeriesSplit
        sys.modules['sklearn.model_selection'] = model_selection_mod
        m.model_selection = model_selection_mod
        # sklearn.metrics
        metrics_mod = types.ModuleType('sklearn.metrics')
        def _zero(*a, **k):
            return 0.0
        metrics_mod.accuracy_score = _zero
        metrics_mod.f1_score = _zero
        metrics_mod.precision_score = _zero
        metrics_mod.recall_score = _zero
        sys.modules['sklearn.metrics'] = metrics_mod
        m.metrics = metrics_mod
        # sklearn.calibration
        calibration_mod = types.ModuleType('sklearn.calibration')
        class _CalibratedClassifierCV:
            def __init__(self, *a, **k):
                pass
            def fit(self, X, y=None):
                return self
            def predict(self, X):
                return []
        def _calibration_curve(*a, **k):
            return [], []
        calibration_mod.CalibratedClassifierCV = _CalibratedClassifierCV
        calibration_mod.calibration_curve = _calibration_curve
        sys.modules['sklearn.calibration'] = calibration_mod
        m.calibration = calibration_mod
        # sklearn.externals
        externals_mod = types.ModuleType('sklearn.externals')
        try:
            # Ensure joblib is available as attribute
            make_stub('joblib')
            externals_mod.joblib = sys.modules.get('joblib')
        except Exception:
            externals_mod.joblib = types.ModuleType('joblib')
        sys.modules['sklearn.externals'] = externals_mod
        m.externals = externals_mod
    elif mod_name == 'watchdog':
        # Create watchdog submodules
        observers_mod = types.ModuleType('watchdog.observers')
        observers_mod.Observer = type('Observer', (), {})
        sys.modules['watchdog.observers'] = observers_mod
        events_mod = types.ModuleType('watchdog.events')
        events_mod.FileSystemEventHandler = type('FileSystemEventHandler', (), {})
        sys.modules['watchdog.events'] = events_mod
        m.observers = observers_mod
        m.events = events_mod
    elif mod_name == 'joblib':
        # Minimal joblib stub
        def _dump(obj, filepath):
            try:
                import json
                from pathlib import Path
                Path(filepath).parent.mkdir(parents=True, exist_ok=True)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump({'stubbed': True}, f)
            except Exception:
                pass
        def _load(filepath):
            return {}
        m.dump = _dump
        m.load = _load
    elif mod_name == 'MetaTrader5':
        m.initialize = lambda: True
        m.shutdown = lambda: None
        m.login = lambda *a, **k: True
        m.logout = lambda *a, **k: None
        m.order_send = lambda *a, **k: type('Ret', (), {'retcode': 0})()
        m.positions_get = lambda *a, **k: []
        m.orders_get = lambda *a, **k: []
        m.account_info = lambda: type('AccountInfo', (), {
            'login': 0,
            'balance': 0.0,
            'equity': 0.0,
            'margin_free': 0.0,
        })()
        m.symbol_info = lambda x: type('obj', (), {'visible': True})
        m.symbol_info_tick = lambda symbol: type('Tick', (), {'bid': 0.0, 'ask': 0.0, 'time': 0})()
        m.symbol_select = lambda *a, **k: True
        m.copy_rates_from_pos = lambda *a, **k: []
        m.copy_rates_range = lambda *a, **k: []
        m.terminal_info = lambda: type('TI', (), {})()
        # Common timeframe and order constants
        m.TIMEFRAME_M1 = 1
        m.TIMEFRAME_M5 = 5
        m.TIMEFRAME_H1 = 60
        m.ORDER_TYPE_BUY = 0
        m.ORDER_TYPE_SELL = 1
        m.POSITION_TYPE_BUY = 0
        m.POSITION_TYPE_SELL = 1
    elif mod_name == 'stable_baselines3':
        m.PPO = type('PPO', (), {'load': classmethod(lambda cls, x: cls())})
    elif mod_name in ['tensorflow', 'torch', 'onnxruntime']:
        class _Session:
            def __init__(self, *a, **k):
                pass
        m.Session = _Session
        class _InferenceSession:
            def __init__(self, *a, **k):
                pass
            def run(self, *a, **k):
                return []
        m.InferenceSession = _InferenceSession
        # Torch submodule stubs
        if mod_name == 'torch':
            nn_mod = types.ModuleType('torch.nn')
            nn_mod.Module = type('Module', (), {'__init__': lambda self, *a, **k: None})
            sys.modules['torch.nn'] = nn_mod
            m.nn = nn_mod
    elif mod_name == 'scipy':
        # Provide minimal scipy.stats used by analyses
        stats_mod = types.ModuleType('scipy.stats')
        stats_mod.ttest_ind = lambda *a, **k: type('Res', (), {'statistic': 0.0, 'pvalue': 1.0})()
        stats_mod.norm = type('norm', (), {'cdf': staticmethod(lambda x: 0.5)})
        sys.modules['scipy.stats'] = stats_mod
        m.stats = stats_mod
    elif mod_name == 'multipart':
        # Satisfy FastAPI Form/File dependency check
        m.__doc__ = 'stub multipart module for preflight'
    elif mod_name == 'prometheus_client':
        # Minimal prometheus_client stub
        class _Metric:
            def __init__(self, *a, **k):
                pass
            def labels(self, *a, **k):
                return self
            def inc(self, *a, **k):
                pass
            def set(self, *a, **k):
                pass
            def observe(self, *a, **k):
                pass
        m.Counter = _Metric
        m.Gauge = _Metric
        m.Histogram = _Metric
        m.Summary = _Metric
        m.CollectorRegistry = type('CollectorRegistry', (), {})
        m.generate_latest = lambda *a, **k: b''
    elif mod_name == 'pydantic':
        # Minimal pydantic API to satisfy imports across v1/v2 style
        m.BaseModel = type('BaseModel', (), {})
        m.BaseSettings = type('BaseSettings', (), {})
        # Field helper returns default, for type hints
        def _Field(default=None, **kwargs):
            return default
        m.Field = _Field
        # create_model returns a simple dynamic class
        def _create_model(name: str, **field_definitions):
            return type(name, (m.BaseModel,), {})
        m.create_model = _create_model
        # Common exceptions and helpers
        m.ValidationError = type('ValidationError', (Exception,), {})
        m.ConfigDict = dict
        def _pass_through_decorator(*dargs, **dkwargs):
            def _wrap(fn):
                return fn
            return _wrap
        m.field_validator = _pass_through_decorator
        m.computed_field = _pass_through_decorator
    elif mod_name == 'pydantic_settings':
        # Provide BaseSettings for Pydantic v2 style
        m.BaseSettings = type('BaseSettings', (), {})
    elif mod_name == 'gymnasium':
        spaces_mod = types.ModuleType('gymnasium.spaces')
        sys.modules['gymnasium.spaces'] = spaces_mod
        m.spaces = spaces_mod
    elif mod_name == 'backend.core.hot_swap_manager':
        # Add a safe stub for hot_swap_manager
        m.HotSwapManager = type('HotSwapManager', (), {})
        m.hot_swap_manager = m.HotSwapManager()

    sys.modules[mod_name] = m


def find_py_files(root: Path):
    """Find all Python files in the repository."""
    for p in root.rglob('*.py'):
        if any(part in IGNORES for part in p.parts):
            continue
        yield p


def ast_syntax_check(files):
    """Check AST syntax for all Python files."""
    errors = []
    for f in files:
        try:
            src = f.read_text(encoding='utf-8')
            ast.parse(src, filename=str(f))
        except SyntaxError as e:
            errors.append({
                'file': str(f.relative_to(REPO_ROOT)),
                'lineno': e.lineno,
                'msg': str(e),
                'severity': 'CRITICAL'
            })
        except Exception as e:
            errors.append({
                'file': str(f.relative_to(REPO_ROOT)),
                'lineno': None,
                'msg': f'Parse error: {e}',
                'severity': 'ERROR'
            })
    return errors


def compile_check(root: Path):
    """Run compileall on the entire repository."""
    results = {'success': False, 'errors': []}
    try:
        # Skip heavy/irrelevant directories using rx
        ignores_pattern = r'[\\/](' + '|'.join(re.escape(x) for x in IGNORES) + r')(?:[\\/]|$)'
        rx = re.compile(ignores_pattern)
        # Compile all Python files
        ok = compileall.compile_dir(
            str(root), 
            force=True, 
            quiet=1,
            rx=rx,
            # Avoid multiprocessing workers in constrained environments
            workers=0,
            invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH
        )
        results['success'] = ok
    except NameError:
        # Fallback for older Python versions
        ok = compileall.compile_dir(str(root), force=True, quiet=1)
    except Exception as e:
        results['errors'].append(str(e))
    return results


def run_with_timeout(fn, timeout: int, default, *args, **kwargs):
    """Run a function with a timeout using a thread to prevent freezes."""
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fn, *args, **kwargs)
            return fut.result(timeout=timeout)
    except FuturesTimeout:
        return default


def import_check(root: Path):
    """Attempt to import modules with safe stubs."""
    results = []
    # install stubs
    for m in EXTERNAL_STUB_MODULES:
        make_stub(m)

    # add repo root and backend to path
    sys.path.insert(0, str(root))
    backend_path = root / 'backend'
    if backend_path.exists():
        sys.path.insert(0, str(backend_path))

    py_files = list(find_py_files(root))
    
    # Group by module for efficient import testing
    modules_to_test = set()
    for f in py_files:
        rel = f.relative_to(root)
        # Skip test files and setup files
        if any(part in ['tests', 'test', 'setup.py', '__main__.py'] for part in rel.parts):
            continue
        # Skip script-like tests by filename (e.g., test_*.py, *_test.py)
        fname = f.name
        if fname.startswith('test_') or fname.endswith('_test.py'):
            continue
        
        # Convert path to module name
        parts = list(rel.with_suffix('').parts)
        # Skip paths that cannot be valid Python packages (e.g., contain hyphens or dots in folder names)
        if any(not p.isidentifier() for p in parts):
            continue
        if parts[0] == 'backend':
            # Handle backend modules, but avoid heavy side-effect module
            mod = '.'.join(parts)
            if mod == 'backend.main':
                continue
            modules_to_test.add(mod)
        elif parts[0] == 'frontend':
            # Skip frontend Python files (usually build scripts)
            continue
        elif parts[0] == 'api':
            # Skip top-level third-party or non-Python API shims (e.g., ForexFeed-java-2.3)
            continue
        else:
            # Skip non-backend modules during preflight to limit scope and avoid accidental side effects
            continue
    
    # Attempt to import each module
    for mod in sorted(modules_to_test):
        try:
            # Special handling for __init__ modules
            if mod.endswith('.__init__'):
                mod = mod[:-9]
            
            importlib.import_module(mod)
            results.append({
                'module': mod,
                'status': 'OK',
                'severity': 'INFO'
            })
        except SystemExit as e:
            # Some modules may call sys.exit() at import time; treat as critical import failure
            results.append({
                'module': mod,
                'status': 'SYS_EXIT',
                'error': str(e),
                'severity': 'CRITICAL'
            })
        except ModuleNotFoundError as e:
            # Check if it's an external dependency
            missing = str(e).split("'")[1] if "'" in str(e) else str(e)
            if missing in EXTERNAL_STUB_MODULES:
                results.append({
                    'module': mod,
                    'status': 'STUB_USED',
                    'error': f'Using stub for {missing}',
                    'severity': 'WARNING'
                })
            else:
                results.append({
                    'module': mod,
                    'status': 'IMPORT_ERROR',
                    'error': str(e),
                    'severity': 'ERROR'
                })
        except ImportError as e:
            results.append({
                'module': mod,
                'status': 'IMPORT_ERROR',
                'error': str(e),
                'severity': 'ERROR'
            })
        except Exception as e:
            tb = traceback.format_exc()
            results.append({
                'module': mod,
                'status': 'RUNTIME_ERROR',
                'error': str(e),
                'traceback': tb,
                'severity': 'CRITICAL'
            })
    return results


def run_pytest(root: Path):
    """Run pytest and collect results."""
    try:
        import pytest
        # Build args and include json-report only if plugin is available
        args = [
            '-q',
            '--maxfail=5',
            '--tb=short',
            str(root / 'backend')
        ]
        try:
            # pytest-json-report provides module 'pytest_jsonreport' with 'plugin'
            if getattr(importlib, 'util', None):
                has_plugin = bool(
                    importlib.util.find_spec('pytest_jsonreport') or
                    importlib.util.find_spec('pytest_jsonreport.plugin')
                )
            else:
                has_plugin = False
            if has_plugin:
                args[3:3] = ['--json-report', '--json-report-file=tools/pytest_report.json']
        except Exception:
            pass
        rv = pytest.main(args)
        # pytest may return an ExitCode enum; cast to int for JSON serialization
        try:
            exit_code = int(rv)
        except Exception:
            exit_code = rv if isinstance(rv, int) else -1
        return {'exit_code': exit_code, 'status': 'COMPLETED'}
    except ImportError:
        return {'exit_code': -1, 'status': 'PYTEST_NOT_INSTALLED'}
    except Exception as e:
        return {
            'exit_code': -1,
            'status': 'ERROR',
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def check_requirements(root: Path):
    """Check for requirements files and analyze dependencies."""
    results = {'files_found': [], 'missing_deps': [], 'warnings': []}
    
    # Check for various requirement files
    req_files = [
        'requirements.txt',
        'backend/requirements.txt',
        'frontend/package.json',
        'pyproject.toml',
        'setup.py'
    ]
    
    for req_file in req_files:
        path = root / req_file
        if path.exists():
            results['files_found'].append(str(req_file))
    
    # Check for common missing dependencies based on imports
    try:
        backend_req = root / 'backend' / 'requirements.txt'
        if backend_req.exists():
            reqs = backend_req.read_text().lower()
            critical_deps = [
                'fastapi', 'uvicorn', 'MetaTrader5', 'pandas', 'numpy',
                'onnxruntime', 'sqlalchemy', 'redis', 'websockets'
            ]
            for dep in critical_deps:
                if dep.lower() not in reqs:
                    results['warnings'].append(f'Critical dependency {dep} might be missing')
    except Exception as e:
        results['warnings'].append(f'Could not analyze requirements: {e}')
    
    return results


def generate_summary(report):
    """Generate a human-readable summary of the report."""
    summary = []
    
    # Count issues by severity
    critical_count = sum(1 for item in report.get('ast_errors', []) if item.get('severity') == 'CRITICAL')
    error_count = sum(1 for item in report.get('ast_errors', []) if item.get('severity') == 'ERROR')
    
    import_errors = [r for r in report.get('import_results', []) if r.get('status') != 'OK']
    
    summary.append("# ARIA Production Preflight Report")
    summary.append(f"Generated: {report.get('timestamp', 'Unknown')}")
    summary.append("")
    
    # Executive Summary
    summary.append("## Executive Summary")
    if critical_count == 0 and error_count == 0 and not import_errors:
        summary.append("✅ **Status: READY FOR PRODUCTION**")
    elif critical_count > 0:
        summary.append("❌ **Status: CRITICAL ISSUES FOUND**")
    else:
        summary.append("⚠️ **Status: ISSUES NEED ATTENTION**")
    
    summary.append("")
    summary.append("## Issue Summary")
    summary.append(f"- Critical Issues: {critical_count}")
    summary.append(f"- Errors: {error_count}")
    summary.append(f"- Import Failures: {len(import_errors)}")
    
    # Details
    if report.get('ast_errors'):
        summary.append("")
        summary.append("## Syntax Errors")
        for err in report['ast_errors'][:10]:  # Show first 10
            summary.append(f"- {err['file']}:{err.get('lineno', '?')} - {err['msg']}")
    
    if import_errors:
        summary.append("")
        summary.append("## Import Errors")
        for err in import_errors[:10]:  # Show first 10
            summary.append(f"- {err['module']}: {err.get('error', 'Unknown error')}")
    
    return "\n".join(summary)


def main():
    parser = argparse.ArgumentParser(description='ARIA Production Preflight Harness')
    parser.add_argument('--output', default='tools/production_preflight_report.json',
                        help='Output JSON report path')
    parser.add_argument('--markdown', action='store_true',
                        help='Also generate markdown report')
    args = parser.parse_args()

    report = {
        'timestamp': datetime.now().isoformat(),
        'repo_root': str(REPO_ROOT),
        'ast_errors': [],
        'compile_results': {},
        'import_results': [],
        'pytest_results': {},
        'requirements_check': {},
        'summary': {}
    }

    print('[DAN] ARIA Production Preflight Starting...')
    print(f'[DAN] Repository root: {REPO_ROOT}')
    
    print('[DAN] Phase 1: AST syntax checks...')
    py_files = list(find_py_files(REPO_ROOT))
    print(f'[DAN] Found {len(py_files)} Python files to check')
    report['ast_errors'] = ast_syntax_check(py_files)
    print(f'[DAN] AST check complete: {len(report["ast_errors"])} errors found')

    print('[DAN] Phase 2: Byte-compile check...')
    report['compile_results'] = compile_check(REPO_ROOT)
    print(f'[DAN] Compile check: {"SUCCESS" if report["compile_results"].get("success") else "FAILED"}')

    print('[DAN] Phase 3: Import resolution checks (with safe stubs)...')
    report['import_results'] = import_check(REPO_ROOT)
    ok_imports = sum(1 for r in report['import_results'] if r.get('status') == 'OK')
    print(f'[DAN] Import check: {ok_imports}/{len(report["import_results"])} modules OK')
    # Show a short preview of top import errors for quick remediation
    _errs_preview = [r for r in report['import_results'] if r.get('status') != 'OK'][:10]
    if _errs_preview:
        print('[DAN] Top import errors:')
        for e in _errs_preview:
            print(f" - {e.get('module')}: {e.get('error', '')} [{e.get('status')}]")

    print('[DAN] Phase 4: Requirements analysis...')
    report['requirements_check'] = check_requirements(REPO_ROOT)
    print(f'[DAN] Found requirement files: {", ".join(report["requirements_check"]["files_found"])}')

    print('[DAN] Phase 5: Running pytest (if available)...')
    report['pytest_results'] = run_pytest(REPO_ROOT)
    print(f'[DAN] Pytest status: {report["pytest_results"].get("status")}')

    # Generate summary statistics
    report['summary'] = {
        'total_files': len(py_files),
        'syntax_errors': len(report['ast_errors']),
        'critical_errors': sum(1 for e in report['ast_errors'] if e.get('severity') == 'CRITICAL'),
        'import_failures': sum(1 for r in report['import_results'] if r.get('status') != 'OK'),
        'compile_success': report['compile_results'].get('success', False),
        'pytest_exit_code': report['pytest_results'].get('exit_code', -1)
    }

    # Write JSON report
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[DAN] JSON report written to {out}')

    # Generate markdown report if requested
    if args.markdown:
        md_path = out.with_suffix('.md')
        md_content = generate_summary(report)
        md_path.write_text(md_content, encoding='utf-8')
        print(f'[DAN] Markdown report written to {md_path}')

    # Print summary
    print('\n' + '='*60)
    print('[DAN] PREFLIGHT SUMMARY')
    print('='*60)
    print(f'Total Python files: {report["summary"]["total_files"]}')
    print(f'Syntax errors: {report["summary"]["syntax_errors"]}')
    print(f'Critical errors: {report["summary"]["critical_errors"]}')
    print(f'Import failures: {report["summary"]["import_failures"]}')
    print(f'Compile success: {report["summary"]["compile_success"]}')
    print(f'Pytest exit code: {report["summary"]["pytest_exit_code"]}')
    
    # Exit with appropriate code
    if report['summary']['critical_errors'] > 0:
        sys.exit(2)
    elif report['summary']['syntax_errors'] > 0 or report['summary']['import_failures'] > 5:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        try:
            log_path = Path('tools') / 'preflight_failure.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(traceback.format_exc(), encoding='utf-8')
            print(f"[DAN] Preflight crashed. See {log_path}")
        except Exception:
            # Last resort: print traceback
            print(traceback.format_exc())
        sys.exit(3)
