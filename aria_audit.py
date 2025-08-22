#!/usr/bin/env python3
"""
ARIA_PRO Institutional-Grade Audit Script
Analyzes backend for orchestration, async patterns, caching, ensemble weights,
risk management, security middleware, and monitoring endpoints.

Usage:
    python aria_audit.py [--output report.md] [--json] [--verbose]
"""

import os
import ast
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re

@dataclass
class AuditFinding:
    category: str
    severity: str  # "info", "warning", "critical"
    title: str
    description: str
    file_path: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None

@dataclass
class AuditReport:
    timestamp: str
    project_path: str
    total_files_scanned: int
    findings: List[AuditFinding]
    summary: Dict[str, Any]
    recommendations: List[str]

class ARIAAuditor:
    def __init__(self, project_path: str, verbose: bool = False):
        self.project_path = Path(project_path)
        self.backend_path = self.project_path / "backend"
        self.verbose = verbose
        self.findings: List[AuditFinding] = []
        self.scanned_files = 0
        
        # Setup logging
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Patterns to detect
        self.async_patterns = [
            r'async\s+def\s+\w+',
            r'await\s+\w+',
            r'asyncio\.gather\(',
            r'asyncio\.create_task\(',
            r'asyncio\.run\(',
        ]
        
        self.caching_patterns = [
            r'redis',
            r'cache',
            r'@lru_cache',
            r'@cached',
            r'memcached',
            r'in_memory',
        ]
        
        self.ensemble_patterns = [
            r'ensemble',
            r'weights',
            r'meta.*learn',
            r'model.*fusion',
            r'weighted.*average',
        ]
        
        self.risk_patterns = [
            r'risk.*engine',
            r'position.*size',
            r'stop.*loss',
            r'max.*drawdown',
            r'risk.*management',
            r'emergency.*stop',
        ]
        
        self.security_patterns = [
            r'TrustedHostMiddleware',
            r'SecurityHeadersMiddleware',
            r'CORS',
            r'CSP',
            r'HSTS',
            r'auth',
            r'token',
        ]
        
        self.monitoring_patterns = [
            r'/monitoring',
            r'/metrics',
            r'/health',
            r'psutil',
            r'logging',
            r'logger',
        ]

    def scan_file(self, file_path: Path) -> None:
        """Scan a single Python file for patterns"""
        try:
            content = file_path.read_text(encoding='utf-8')
            self.scanned_files += 1
            
            # Parse AST for deeper analysis
            try:
                tree = ast.parse(content)
                self._analyze_ast(tree, file_path, content)
            except SyntaxError:
                self.logger.warning(f"Syntax error in {file_path}, skipping AST analysis")
            
            # Pattern matching analysis
            self._analyze_patterns(content, file_path)
            
        except Exception as e:
            self.logger.error(f"Error scanning {file_path}: {e}")

    def _analyze_ast(self, tree: ast.AST, file_path: Path, content: str) -> None:
        """Analyze AST for structural patterns"""
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            # Check for async function definitions
            if isinstance(node, ast.AsyncFunctionDef):
                self._add_finding(
                    "orchestration",
                    "info",
                    f"Async function found: {node.name}",
                    f"Found async function '{node.name}' which supports concurrent execution",
                    str(file_path),
                    node.lineno,
                    "Ensure proper error handling and cancellation for async operations"
                )
            
            # Check for asyncio.gather usage
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (isinstance(node.func.value, ast.Name) and 
                    node.func.value.id == 'asyncio' and 
                    node.func.attr == 'gather'):
                    self._add_finding(
                        "orchestration",
                        "info",
                        "Parallel async execution found",
                        "asyncio.gather() enables concurrent model inference",
                        str(file_path),
                        node.lineno,
                        "Consider adding timeout and error handling for gather operations"
                    )
            
            # Check for class definitions (potential orchestrators)
            if isinstance(node, ast.ClassDef):
                class_name = node.name.lower()
                if any(keyword in class_name for keyword in ['manager', 'orchestrator', 'ensemble', 'engine']):
                    self._add_finding(
                        "orchestration",
                        "info",
                        f"Orchestration class found: {node.name}",
                        f"Class '{node.name}' appears to be an orchestration component",
                        str(file_path),
                        node.lineno,
                        "Verify this class implements proper lifecycle management and error handling"
                    )
            
            # Check for weight/ensemble assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and 'weight' in target.id.lower():
                        self._add_finding(
                            "ensemble",
                            "info",
                            f"Weight assignment found: {target.id}",
                            "Found weight assignment which may be part of ensemble logic",
                            str(file_path),
                            node.lineno,
                            "Ensure weights are normalized and configurable"
                        )

    def _analyze_patterns(self, content: str, file_path: Path) -> None:
        """Analyze content using regex patterns"""
        lines = content.split('\n')
        
        # Check async patterns
        for i, line in enumerate(lines, 1):
            for pattern in self.async_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    self._add_finding(
                        "async",
                        "info",
                        "Async pattern detected",
                        f"Found async pattern: {pattern}",
                        str(file_path),
                        i,
                        code_snippet=line.strip()
                    )
        
        # Check caching patterns
        cache_found = False
        for i, line in enumerate(lines, 1):
            for pattern in self.caching_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    cache_found = True
                    self._add_finding(
                        "caching",
                        "info",
                        "Caching mechanism found",
                        f"Found caching pattern: {pattern}",
                        str(file_path),
                        i,
                        code_snippet=line.strip()
                    )
        
        if not cache_found and 'signal_generator' in str(file_path).lower():
            self._add_finding(
                "caching",
                "warning",
                "No caching found in signal generator",
                "Signal generators should implement caching for performance",
                str(file_path),
                suggestion="Add Redis or in-memory caching for model predictions and market data"
            )
        
        # Check ensemble patterns
        ensemble_found = False
        for i, line in enumerate(lines, 1):
            for pattern in self.ensemble_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    ensemble_found = True
                    self._add_finding(
                        "ensemble",
                        "info",
                        "Ensemble logic found",
                        f"Found ensemble pattern: {pattern}",
                        str(file_path),
                        i,
                        code_snippet=line.strip()
                    )
        
        # Check risk management
        risk_found = False
        for i, line in enumerate(lines, 1):
            for pattern in self.risk_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    risk_found = True
                    self._add_finding(
                        "risk",
                        "info",
                        "Risk management found",
                        f"Found risk pattern: {pattern}",
                        str(file_path),
                        i,
                        code_snippet=line.strip()
                    )
        
        # Check security patterns
        security_found = False
        for i, line in enumerate(lines, 1):
            for pattern in self.security_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    security_found = True
                    self._add_finding(
                        "security",
                        "info",
                        "Security mechanism found",
                        f"Found security pattern: {pattern}",
                        str(file_path),
                        i,
                        code_snippet=line.strip()
                    )
        
        # Check monitoring patterns
        monitoring_found = False
        for i, line in enumerate(lines, 1):
            for pattern in self.monitoring_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    monitoring_found = True
                    self._add_finding(
                        "monitoring",
                        "info",
                        "Monitoring mechanism found",
                        f"Found monitoring pattern: {pattern}",
                        str(file_path),
                        i,
                        code_snippet=line.strip()
                    )

    def _add_finding(self, category: str, severity: str, title: str, description: str, 
                    file_path: str, line_number: Optional[int] = None, 
                    suggestion: Optional[str] = None, code_snippet: Optional[str] = None) -> None:
        """Add a finding to the audit results"""
        finding = AuditFinding(
            category=category,
            severity=severity,
            title=title,
            description=description,
            file_path=file_path,
            line_number=line_number,
            suggestion=suggestion,
            code_snippet=code_snippet
        )
        self.findings.append(finding)

    def audit_backend(self) -> None:
        """Main audit function for backend directory"""
        if not self.backend_path.exists():
            self.logger.error(f"Backend path not found: {self.backend_path}")
            return
        
        self.logger.info(f"Starting audit of {self.backend_path}")
        
        # Scan all Python files
        for py_file in self.backend_path.rglob("*.py"):
            if self.verbose:
                self.logger.debug(f"Scanning {py_file}")
            self.scan_file(py_file)
        
        # Additional specific checks
        self._check_main_py()
        self._check_config_management()
        self._check_model_orchestration()
        self._check_risk_implementation()
        self._check_monitoring_endpoints()
        self._check_security_middleware()

    def _check_main_py(self) -> None:
        """Check main.py for proper app structure"""
        main_py = self.backend_path / "main.py"
        if not main_py.exists():
            self._add_finding(
                "orchestration",
                "critical",
                "main.py not found",
                "No main.py found in backend directory",
                str(main_py),
                suggestion="Create main.py with FastAPI app initialization"
            )
            return
        
        content = main_py.read_text()
        
        # Check for middleware registration
        if "TrustedHostMiddleware" not in content:
            self._add_finding(
                "security",
                "warning",
                "TrustedHostMiddleware not found",
                "TrustedHostMiddleware should be configured for production security",
                str(main_py),
                suggestion="Add TrustedHostMiddleware to main.py"
            )
        
        if "CORSMiddleware" not in content:
            self._add_finding(
                "security",
                "warning",
                "CORSMiddleware not configured",
                "CORS should be properly configured for cross-origin requests",
                str(main_py),
                suggestion="Add CORSMiddleware with explicit origins"
            )

    def _check_config_management(self) -> None:
        """Check configuration management"""
        config_files = list(self.backend_path.rglob("*config*.py"))
        
        if not config_files:
            self._add_finding(
                "configuration",
                "warning",
                "No configuration management found",
                "No centralized configuration files detected",
                str(self.backend_path),
                suggestion="Implement centralized configuration with Pydantic Settings"
            )
        
        for config_file in config_files:
            content = config_file.read_text()
            if "BaseSettings" in content or "pydantic" in content:
                self._add_finding(
                    "configuration",
                    "info",
                    "Pydantic configuration found",
                    f"Found Pydantic-based configuration in {config_file.name}",
                    str(config_file),
                    suggestion="Ensure all environment variables are properly typed and documented"
                )

    def _check_model_orchestration(self) -> None:
        """Check for model orchestration patterns"""
        orchestration_files = []
        
        for py_file in self.backend_path.rglob("*.py"):
            content = py_file.read_text()
            if any(keyword in content.lower() for keyword in ['ensemble', 'meta_learn', 'model_manager', 'orchestrat']):
                orchestration_files.append(py_file)
        
        if not orchestration_files:
            self._add_finding(
                "orchestration",
                "warning",
                "No model orchestration found",
                "No clear model orchestration or ensemble management detected",
                str(self.backend_path),
                suggestion="Implement ModelManager or EnsembleOrchestrator for coordinated model execution"
            )
        
        # Check for async model execution
        async_model_execution = False
        for py_file in orchestration_files:
            content = py_file.read_text()
            if "asyncio.gather" in content and "model" in content.lower():
                async_model_execution = True
                break
        
        if not async_model_execution:
            self._add_finding(
                "orchestration",
                "warning",
                "No async model execution found",
                "Models should be executed concurrently for optimal performance",
                str(self.backend_path),
                suggestion="Use asyncio.gather() for parallel model inference"
            )

    def _check_risk_implementation(self) -> None:
        """Check risk management implementation"""
        risk_files = list(self.backend_path.rglob("*risk*.py"))
        
        if not risk_files:
            self._add_finding(
                "risk",
                "critical",
                "No risk management found",
                "No dedicated risk management modules detected",
                str(self.backend_path),
                suggestion="Implement RiskEngine with position sizing, stop losses, and drawdown limits"
            )
            return
        
        for risk_file in risk_files:
            content = risk_file.read_text()
            
            # Check for essential risk components
            risk_components = [
                ("position_size", "Position sizing logic"),
                ("stop_loss", "Stop loss implementation"),
                ("max_drawdown", "Maximum drawdown protection"),
                ("emergency_stop", "Emergency stop mechanism")
            ]
            
            for component, description in risk_components:
                if component not in content.lower():
                    self._add_finding(
                        "risk",
                        "warning",
                        f"Missing {description}",
                        f"Risk file {risk_file.name} missing {description}",
                        str(risk_file),
                        suggestion=f"Implement {description} in risk management"
                    )

    def _check_monitoring_endpoints(self) -> None:
        """Check for monitoring and observability"""
        monitoring_endpoints = []
        
        for py_file in self.backend_path.rglob("*.py"):
            content = py_file.read_text()
            
            # Look for monitoring routes
            if "@router.get" in content and any(endpoint in content for endpoint in ["/health", "/metrics", "/monitoring"]):
                monitoring_endpoints.append(py_file)
        
        if not monitoring_endpoints:
            self._add_finding(
                "monitoring",
                "warning",
                "No monitoring endpoints found",
                "No health check or metrics endpoints detected",
                str(self.backend_path),
                suggestion="Add /health, /metrics, and /monitoring endpoints for observability"
            )
        
        # Check for structured logging
        logging_found = False
        for py_file in self.backend_path.rglob("*.py"):
            content = py_file.read_text()
            if "logging.getLogger" in content:
                logging_found = True
                break
        
        if not logging_found:
            self._add_finding(
                "monitoring",
                "warning",
                "No structured logging found",
                "No structured logging implementation detected",
                str(self.backend_path),
                suggestion="Implement structured logging with proper log levels and formatting"
            )

    def _check_security_middleware(self) -> None:
        """Check security middleware implementation"""
        main_py = self.backend_path / "main.py"
        if not main_py.exists():
            return
        
        content = main_py.read_text()
        
        security_checks = [
            ("SecurityHeadersMiddleware", "Security headers middleware"),
            ("TrustedHostMiddleware", "Trusted host middleware"),
            ("HSTS", "HTTP Strict Transport Security"),
            ("CSP", "Content Security Policy"),
            ("X-Frame-Options", "Frame options header")
        ]
        
        for check, description in security_checks:
            if check not in content:
                self._add_finding(
                    "security",
                    "warning",
                    f"Missing {description}",
                    f"Security feature {description} not implemented",
                    str(main_py),
                    suggestion=f"Implement {description} for production security"
                )

    def generate_report(self) -> AuditReport:
        """Generate comprehensive audit report"""
        # Categorize findings
        categories = {}
        severities = {"info": 0, "warning": 0, "critical": 0}
        
        for finding in self.findings:
            categories[finding.category] = categories.get(finding.category, 0) + 1
            severities[finding.severity] += 1
        
        # Generate recommendations based on findings
        recommendations = self._generate_recommendations()
        
        # Create summary
        summary = {
            "total_findings": len(self.findings),
            "categories": categories,
            "severities": severities,
            "files_scanned": self.scanned_files,
            "critical_issues": severities["critical"],
            "warnings": severities["warning"],
            "info_items": severities["info"]
        }
        
        return AuditReport(
            timestamp=datetime.now().isoformat(),
            project_path=str(self.project_path),
            total_files_scanned=self.scanned_files,
            findings=self.findings,
            summary=summary,
            recommendations=recommendations
        )

    def _generate_recommendations(self) -> List[str]:
        """Generate prioritized recommendations"""
        recommendations = []
        
        # Count critical issues by category
        critical_by_category = {}
        warning_by_category = {}
        
        for finding in self.findings:
            if finding.severity == "critical":
                critical_by_category[finding.category] = critical_by_category.get(finding.category, 0) + 1
            elif finding.severity == "warning":
                warning_by_category[finding.category] = warning_by_category.get(finding.category, 0) + 1
        
        # Priority recommendations based on critical issues
        if critical_by_category.get("risk", 0) > 0:
            recommendations.append("🚨 CRITICAL: Implement comprehensive risk management with RiskEngine, position sizing, and emergency stops")
        
        if critical_by_category.get("orchestration", 0) > 0:
            recommendations.append("🚨 CRITICAL: Set up proper application orchestration with main.py and service management")
        
        # High-priority warnings
        if warning_by_category.get("orchestration", 0) > 0:
            recommendations.append("⚠️ HIGH: Implement async model orchestration with asyncio.gather() for parallel inference")
        
        if warning_by_category.get("caching", 0) > 0:
            recommendations.append("⚠️ HIGH: Add caching layer (Redis/in-memory) for model predictions and market data")
        
        if warning_by_category.get("security", 0) > 0:
            recommendations.append("⚠️ HIGH: Strengthen security with proper middleware, CORS, and headers")
        
        if warning_by_category.get("monitoring", 0) > 0:
            recommendations.append("⚠️ MEDIUM: Add comprehensive monitoring with /health, /metrics endpoints and structured logging")
        
        # Performance recommendations
        if warning_by_category.get("ensemble", 0) > 0:
            recommendations.append("💡 OPTIMIZATION: Implement dynamic ensemble weights with meta-learning")
        
        # General recommendations
        recommendations.extend([
            "📊 Add performance profiling for model inference latency",
            "🔧 Implement configuration validation and environment-specific settings",
            "📝 Add comprehensive API documentation with OpenAPI/Swagger",
            "🧪 Create integration tests for critical trading paths",
            "🔄 Set up automated model retraining and deployment pipeline"
        ])
        
        return recommendations

def format_markdown_report(report: AuditReport) -> str:
    """Format audit report as Markdown"""
    md = f"""# ARIA PRO Institutional Trading System - Audit Report

**Generated:** {report.timestamp}  
**Project Path:** `{report.project_path}`  
**Files Scanned:** {report.total_files_scanned}

## 📊 Executive Summary

| Metric | Count |
|--------|-------|
| **Total Findings** | {report.summary['total_findings']} |
| **Critical Issues** | {report.summary['critical_issues']} |
| **Warnings** | {report.summary['warnings']} |
| **Info Items** | {report.summary['info_items']} |

### Findings by Category
"""
    
    for category, count in report.summary['categories'].items():
        md += f"- **{category.title()}:** {count} findings\n"
    
    md += f"""
## 🎯 Priority Recommendations

"""
    for i, rec in enumerate(report.recommendations[:10], 1):
        md += f"{i}. {rec}\n"
    
    md += f"""
## 🔍 Detailed Findings

"""
    
    # Group findings by category
    by_category = {}
    for finding in report.findings:
        if finding.category not in by_category:
            by_category[finding.category] = []
        by_category[finding.category].append(finding)
    
    for category, findings in by_category.items():
        md += f"### {category.title()} ({len(findings)} findings)\n\n"
        
        for finding in findings:
            severity_icon = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}[finding.severity]
            md += f"#### {severity_icon} {finding.title}\n\n"
            md += f"**File:** `{finding.file_path}`"
            if finding.line_number:
                md += f" (Line {finding.line_number})"
            md += f"\n\n**Description:** {finding.description}\n\n"
            
            if finding.code_snippet:
                md += f"**Code:**\n```python\n{finding.code_snippet}\n```\n\n"
            
            if finding.suggestion:
                md += f"**Suggestion:** {finding.suggestion}\n\n"
            
            md += "---\n\n"
    
    md += f"""
## 🏗️ Architecture Improvements

Based on the audit findings, here are architectural improvements for ARIA PRO:

### 1. Model Orchestration & Pipeline
- Implement `ModelOrchestrator` class for coordinated AI model execution
- Use `asyncio.gather()` for parallel model inference (target: <100ms total latency)
- Add model health checks and fallback mechanisms
- Implement model versioning and A/B testing framework

### 2. Caching & Performance
- Deploy Redis cluster for model prediction caching
- Implement in-memory LRU cache for market data (last 1000 ticks)
- Add model artifact caching to avoid repeated loading
- Optimize data serialization with MessagePack or Protocol Buffers

### 3. Risk Management Enhancement
- Centralize risk rules in `RiskEngine` with real-time monitoring
- Implement dynamic position sizing based on volatility regimes
- Add correlation-based portfolio risk management
- Create emergency circuit breakers with automatic recovery

### 4. Security Hardening
- Enable all security middleware in production
- Implement JWT-based API authentication
- Add rate limiting per client/endpoint
- Set up WAF rules for API protection

### 5. Monitoring & Observability
- Deploy Prometheus + Grafana for metrics visualization
- Add distributed tracing with OpenTelemetry
- Implement structured logging with ELK stack
- Create real-time alerting for trading anomalies

### 6. Ensemble Intelligence
- Implement meta-learning for dynamic model weights
- Add regime detection for model selection
- Create confidence-based ensemble decisions
- Deploy online learning for weight adaptation

## 📈 Performance Targets

| Metric | Current | Target | Priority |
|--------|---------|--------|----------|
| Model Inference | ~500ms | <100ms | High |
| API Response Time | ~200ms | <50ms | High |
| Memory Usage | ~2GB | <1GB | Medium |
| Cache Hit Rate | 0% | >90% | High |
| System Uptime | 95% | 99.9% | Critical |

## 🚀 Implementation Roadmap

### Phase 1: Critical Fixes (Week 1)
- Fix all critical security and risk management issues
- Implement basic monitoring endpoints
- Add error handling and logging

### Phase 2: Performance (Week 2-3)
- Deploy caching layer
- Implement async model orchestration  
- Optimize database queries and API responses

### Phase 3: Intelligence (Week 4-6)
- Deploy ensemble meta-learning
- Add regime detection and adaptation
- Implement advanced risk management

### Phase 4: Scale & Monitor (Week 7-8)
- Full monitoring and alerting setup
- Performance optimization and tuning
- Load testing and capacity planning

---

*This audit was generated by ARIA Institutional Audit Framework v1.0*
"""
    
    return md

def main():
    parser = argparse.ArgumentParser(description="ARIA PRO Institutional Audit Script")
    parser.add_argument("--project-path", default=".", help="Path to ARIA_PRO project")
    parser.add_argument("--output", default="aria_audit_report.md", help="Output report file")
    parser.add_argument("--json", action="store_true", help="Also output JSON report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Initialize auditor
    auditor = ARIAAuditor(args.project_path, args.verbose)
    
    # Run audit
    print("🔍 Starting ARIA PRO institutional audit...")
    auditor.audit_backend()
    
    # Generate report
    print("📊 Generating audit report...")
    report = auditor.generate_report()
    
    # Output Markdown report
    markdown_report = format_markdown_report(report)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    
    print(f"✅ Audit complete! Report saved to: {args.output}")
    
    # Output JSON if requested
    if args.json:
        json_output = args.output.replace('.md', '.json')
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        print(f"📄 JSON report saved to: {json_output}")
    
    # Print summary
    print(f"\n📋 Audit Summary:")
    print(f"   Files Scanned: {report.total_files_scanned}")
    print(f"   Total Findings: {report.summary['total_findings']}")
    print(f"   Critical Issues: {report.summary['critical_issues']}")
    print(f"   Warnings: {report.summary['warnings']}")
    print(f"   Info Items: {report.summary['info_items']}")
    
    if report.summary['critical_issues'] > 0:
        print(f"\n🚨 ATTENTION: {report.summary['critical_issues']} critical issues found!")
        print("   Review the report immediately for security and risk management fixes.")

if __name__ == "__main__":
    main()
