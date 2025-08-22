# ARIA Production Preflight Report
Generated: 2025-08-22T03:02:20.969606

## Executive Summary
⚠️ **Status: ISSUES NEED ATTENTION**

## Issue Summary
- Critical Issues: 0
- Errors: 0
- Import Failures: 51

## Import Errors
- backend.automation.cicd_pipeline: 'Observer' object has no attribute 'schedule'
- backend.core.data_connector: cannot import name 'is_datetime64tz_dtype' from 'pandas.api.types' (unknown location)
- backend.core.data_sanity: cannot import name 'is_datetime64tz_dtype' from 'pandas.api.types' (unknown location)
- backend.core.model_loader: 'Observer' object has no attribute 'schedule'
- backend.core.phase3_orchestrator: No module named 'pydantic.version'; 'pydantic' is not a package
- backend.core.training_connector: cannot import name 'is_datetime64tz_dtype' from 'pandas.api.types' (unknown location)
- backend.ensemble.advanced_ensemble_optimizer: 'Observer' object has no attribute 'schedule'
- backend.monitoring.llm_monitor: No module named 'pydantic.version'; 'pydantic' is not a package
- backend.routes.account: No module named 'pydantic.version'; 'pydantic' is not a package
- backend.routes.analytics: No module named 'pydantic.version'; 'pydantic' is not a package