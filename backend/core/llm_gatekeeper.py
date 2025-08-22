"""
LLM Gatekeeper with Guardrails
Provides safe LLM usage with prompt injection protection and output validation
"""

import logging
import time
import hashlib
import re
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class SecurityThreat:
    """Detected security threat"""
    threat_type: str  # 'injection', 'jailbreak', 'exfiltration', 'manipulation'
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    evidence: str
    timestamp: float = field(default_factory=time.time)
    blocked: bool = True


@dataclass
class LLMRequest:
    """Validated LLM request"""
    prompt: str
    context: Dict[str, Any]
    purpose: str  # 'analysis', 'signal', 'report', 'explanation'
    max_tokens: int = 500
    temperature: float = 0.3
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class LLMResponse:
    """Validated LLM response"""
    content: str
    filtered: bool
    threats_detected: List[SecurityThreat]
    confidence: float
    tokens_used: int
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


class PromptSanitizer:
    """Sanitizes prompts to prevent injection attacks"""
    
    def __init__(self):
        # Dangerous patterns that indicate injection attempts
        self.injection_patterns = [
            # Direct instruction overrides
            r"ignore\s+(previous|all|above)",
            r"disregard\s+(instructions|rules)",
            r"forget\s+(everything|all)",
            r"new\s+instructions?:",
            r"system\s*:\s*",
            r"admin\s*mode",
            r"developer\s*mode",
            
            # Role manipulation
            r"you\s+are\s+now",
            r"act\s+as\s+if",
            r"pretend\s+to\s+be",
            r"roleplay\s+as",
            r"simulate\s+being",
            
            # Data exfiltration attempts
            r"list\s+all\s+(api|keys|credentials)",
            r"show\s+me\s+(passwords?|secrets?)",
            r"reveal\s+(internal|private|confidential)",
            r"dump\s+(database|memory|config)",
            
            # Code execution attempts
            r"execute\s+(code|command|script)",
            r"eval\s*\(",
            r"os\s*\.\s*system",
            r"subprocess",
            r"__import__",
            
            # Bypass attempts
            r"base64\s*decode",
            r"rot13",
            r"hex\s*decode",
            r"\\x[0-9a-f]{2}",  # Hex escapes
            r"\\u[0-9a-f]{4}",  # Unicode escapes
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.injection_patterns
        ]
        
        # Safe character whitelist for trading context
        self.safe_chars = set(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " .,;:!?-_+*/=()[]{}'\"\n\t"
            "$€£¥%@#&"
        )
        
    def sanitize(self, prompt: str) -> Tuple[str, List[SecurityThreat]]:
        """
        Sanitize prompt and detect threats
        Returns: (sanitized_prompt, threats)
        """
        threats = []
        
        # Check for injection patterns
        for pattern in self.compiled_patterns:
            if pattern.search(prompt):
                threats.append(SecurityThreat(
                    threat_type='injection',
                    severity='high',
                    description=f'Potential injection pattern detected',
                    evidence=pattern.pattern[:50],
                    blocked=True
                ))
                
        # Check for suspicious character sequences
        suspicious_chars = set(prompt) - self.safe_chars
        if suspicious_chars:
            threats.append(SecurityThreat(
                threat_type='manipulation',
                severity='medium',
                description='Suspicious characters detected',
                evidence=str(suspicious_chars)[:100],
                blocked=False
            ))
            
        # Check for excessive length (possible buffer overflow attempt)
        if len(prompt) > 5000:
            threats.append(SecurityThreat(
                threat_type='manipulation',
                severity='medium',
                description='Excessive prompt length',
                evidence=f'Length: {len(prompt)}',
                blocked=True
            ))
            
        # Check for repeated patterns (possible DoS)
        if self._has_excessive_repetition(prompt):
            threats.append(SecurityThreat(
                threat_type='manipulation',
                severity='low',
                description='Excessive repetition detected',
                evidence='Repeated patterns found',
                blocked=False
            ))
            
        # Sanitize the prompt
        sanitized = prompt
        
        # Remove any detected injection patterns
        for pattern in self.compiled_patterns:
            sanitized = pattern.sub('[BLOCKED]', sanitized)
            
        # Remove non-safe characters
        if suspicious_chars:
            for char in suspicious_chars:
                sanitized = sanitized.replace(char, '')
                
        # Truncate if too long
        if len(sanitized) > 2000:
            sanitized = sanitized[:2000] + '... [TRUNCATED]'
            
        return sanitized, threats
        
    def _has_excessive_repetition(self, text: str, threshold: int = 10) -> bool:
        """Check for excessive repetition in text"""
        # Check for repeated words
        words = text.split()
        if len(words) > 0:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            max_repetition = max(word_counts.values())
            if max_repetition > threshold:
                return True
                
        # Check for repeated characters
        for i in range(len(text) - threshold):
            if text[i:i+3] * (threshold // 3) in text:
                return True
                
        return False


class OutputValidator:
    """Validates LLM outputs for safety and quality"""
    
    def __init__(self):
        # Forbidden output patterns
        self.forbidden_patterns = [
            # API keys and secrets
            r"api[_\s]?key\s*[:=]\s*[\w\-]+",
            r"secret\s*[:=]\s*[\w\-]+",
            r"password\s*[:=]\s*[\w\-]+",
            r"token\s*[:=]\s*[\w\-]+",
            
            # Personal information
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{16}\b",  # Credit card
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Email
            
            # Code execution
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",  # Event handlers
            
            # SQL injection
            r";\s*(DROP|DELETE|INSERT|UPDATE)\s+",
            r"--\s*$",  # SQL comment
            
            # System paths
            r"[cC]:\\",
            r"/etc/",
            r"/usr/",
            r"~/",
        ]
        
        self.compiled_forbidden = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for pattern in self.forbidden_patterns
        ]
        
        # Quality checks
        self.min_coherence_score = 0.3
        self.max_hallucination_score = 0.7
        
    def validate(self, output: str, context: Dict) -> Tuple[str, List[SecurityThreat]]:
        """
        Validate output for safety and quality
        Returns: (filtered_output, threats)
        """
        threats = []
        
        # Check for forbidden patterns
        for pattern in self.compiled_forbidden:
            if pattern.search(output):
                threats.append(SecurityThreat(
                    threat_type='exfiltration',
                    severity='critical',
                    description='Forbidden pattern in output',
                    evidence=pattern.pattern[:50],
                    blocked=True
                ))
                
        # Check coherence
        coherence = self._calculate_coherence(output)
        if coherence < self.min_coherence_score:
            threats.append(SecurityThreat(
                threat_type='manipulation',
                severity='low',
                description='Low coherence output',
                evidence=f'Score: {coherence:.2f}',
                blocked=False
            ))
            
        # Check for hallucinations
        hallucination_score = self._check_hallucination(output, context)
        if hallucination_score > self.max_hallucination_score:
            threats.append(SecurityThreat(
                threat_type='manipulation',
                severity='medium',
                description='Potential hallucination detected',
                evidence=f'Score: {hallucination_score:.2f}',
                blocked=False
            ))
            
        # Filter the output
        filtered = output
        
        # Remove forbidden patterns
        for pattern in self.compiled_forbidden:
            filtered = pattern.sub('[REDACTED]', filtered)
            
        # Add warning for low quality
        if coherence < self.min_coherence_score:
            filtered = f"[LOW CONFIDENCE] {filtered}"
            
        return filtered, threats
        
    def _calculate_coherence(self, text: str) -> float:
        """Calculate text coherence score (0-1)"""
        if not text:
            return 0.0
            
        # Simple heuristics for coherence
        score = 1.0
        
        # Check sentence structure
        sentences = text.split('.')
        if len(sentences) < 2:
            score *= 0.7
            
        # Check word variety
        words = text.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            score *= unique_ratio
            
        # Check for gibberish (repeated characters)
        if re.search(r'(.)\1{5,}', text):
            score *= 0.5
            
        return min(1.0, max(0.0, score))
        
    def _check_hallucination(self, output: str, context: Dict) -> float:
        """Check for potential hallucinations (0-1, higher = more likely)"""
        score = 0.0
        
        # Check for numbers not in context
        output_numbers = set(re.findall(r'\b\d+\.?\d*\b', output))
        context_str = str(context)
        context_numbers = set(re.findall(r'\b\d+\.?\d*\b', context_str))
        
        if output_numbers - context_numbers:
            score += 0.3
            
        # Check for ungrounded claims
        confidence_words = ['definitely', 'certainly', 'absolutely', 'guaranteed']
        for word in confidence_words:
            if word in output.lower():
                score += 0.2
                
        # Check for fictional entities
        if re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', output):
            # Proper names not in context
            names_in_output = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', output)
            for name in names_in_output:
                if name not in context_str:
                    score += 0.1
                    
        return min(1.0, score)


class RateLimiter:
    """Rate limiting for LLM requests"""
    
    def __init__(self):
        self.requests_per_minute = 30
        self.requests_per_hour = 500
        self.requests_per_day = 5000
        
        self.minute_window = deque(maxlen=self.requests_per_minute)
        self.hour_window = deque(maxlen=self.requests_per_hour)
        self.day_window = deque(maxlen=self.requests_per_day)
        
    def check_limit(self, user_id: str = 'default') -> Tuple[bool, str]:
        """
        Check if request is within rate limits
        Returns: (allowed, reason)
        """
        now = time.time()
        
        # Clean old entries
        minute_ago = now - 60
        hour_ago = now - 3600
        day_ago = now - 86400
        
        # Check minute limit
        recent_minute = [t for t in self.minute_window if t > minute_ago]
        if len(recent_minute) >= self.requests_per_minute:
            return False, f"Rate limit exceeded: {self.requests_per_minute}/min"
            
        # Check hour limit
        recent_hour = [t for t in self.hour_window if t > hour_ago]
        if len(recent_hour) >= self.requests_per_hour:
            return False, f"Rate limit exceeded: {self.requests_per_hour}/hour"
            
        # Check day limit
        recent_day = [t for t in self.day_window if t > day_ago]
        if len(recent_day) >= self.requests_per_day:
            return False, f"Rate limit exceeded: {self.requests_per_day}/day"
            
        # Record request
        self.minute_window.append(now)
        self.hour_window.append(now)
        self.day_window.append(now)
        
        return True, "OK"


class LLMGatekeeper:
    """Main LLM Gatekeeper with comprehensive security"""
    
    def __init__(self):
        self.sanitizer = PromptSanitizer()
        self.validator = OutputValidator()
        self.rate_limiter = RateLimiter()
        
        # Audit log
        self.audit_log = deque(maxlen=1000)
        self.threat_log = deque(maxlen=100)
        
        # Statistics
        self.total_requests = 0
        self.blocked_requests = 0
        self.filtered_outputs = 0
        
        # Approved purposes for LLM usage
        self.approved_purposes = {
            'analysis': 'Market analysis and interpretation',
            'signal': 'Trading signal generation',
            'report': 'Performance reporting',
            'explanation': 'Trade explanation and reasoning',
            'sentiment': 'News and sentiment analysis'
        }
        
        # Context validation rules
        self.required_context = {
            'analysis': ['symbol', 'timeframe'],
            'signal': ['symbol', 'price', 'indicators'],
            'report': ['period', 'metrics'],
            'explanation': ['trade_id', 'decision'],
            'sentiment': ['text', 'source']
        }
        
    async def process_request(self, 
                            prompt: str,
                            purpose: str,
                            context: Dict,
                            user_id: Optional[str] = None) -> LLMResponse:
        """
        Process LLM request with full security pipeline
        """
        start_time = time.time()
        self.total_requests += 1
        
        # Create request object
        request = LLMRequest(
            prompt=prompt,
            context=context,
            purpose=purpose,
            user_id=user_id
        )
        
        # Validate purpose
        if purpose not in self.approved_purposes:
            self.blocked_requests += 1
            return LLMResponse(
                content="Invalid purpose for LLM usage",
                filtered=True,
                threats_detected=[SecurityThreat(
                    threat_type='manipulation',
                    severity='high',
                    description='Unapproved purpose',
                    evidence=purpose
                )],
                confidence=0.0,
                tokens_used=0,
                latency_ms=0
            )
            
        # Validate required context
        if purpose in self.required_context:
            missing = [k for k in self.required_context[purpose] if k not in context]
            if missing:
                return LLMResponse(
                    content=f"Missing required context: {missing}",
                    filtered=True,
                    threats_detected=[],
                    confidence=0.0,
                    tokens_used=0,
                    latency_ms=0
                )
                
        # Rate limiting
        allowed, reason = self.rate_limiter.check_limit(user_id or 'default')
        if not allowed:
            self.blocked_requests += 1
            return LLMResponse(
                content=f"Rate limit exceeded: {reason}",
                filtered=True,
                threats_detected=[],
                confidence=0.0,
                tokens_used=0,
                latency_ms=0
            )
            
        # Sanitize prompt
        sanitized_prompt, prompt_threats = self.sanitizer.sanitize(prompt)
        
        # Block if critical threats detected
        critical_threats = [t for t in prompt_threats if t.severity in ['high', 'critical']]
        if critical_threats:
            self.blocked_requests += 1
            self._log_threats(critical_threats)
            
            return LLMResponse(
                content="Request blocked due to security threats",
                filtered=True,
                threats_detected=critical_threats,
                confidence=0.0,
                tokens_used=0,
                latency_ms=(time.time() - start_time) * 1000
            )
            
        # Build safe prompt with context
        safe_prompt = self._build_safe_prompt(sanitized_prompt, purpose, context)
        
        # Simulate LLM call (replace with actual LLM integration)
        llm_output = await self._call_llm(safe_prompt)
        
        # Validate output
        filtered_output, output_threats = self.validator.validate(llm_output, context)
        
        if filtered_output != llm_output:
            self.filtered_outputs += 1
            
        # Combine all threats
        all_threats = prompt_threats + output_threats
        
        # Log if threats detected
        if all_threats:
            self._log_threats(all_threats)
            
        # Calculate confidence based on threats
        confidence = self._calculate_confidence(all_threats)
        
        # Audit log
        self._audit_log(request, filtered_output, all_threats)
        
        return LLMResponse(
            content=filtered_output,
            filtered=filtered_output != llm_output,
            threats_detected=all_threats,
            confidence=confidence,
            tokens_used=len(safe_prompt.split()) + len(filtered_output.split()),
            latency_ms=(time.time() - start_time) * 1000
        )
        
    def _build_safe_prompt(self, prompt: str, purpose: str, context: Dict) -> str:
        """Build safe prompt with proper context and constraints"""
        
        safe_prompt = f"""
[SYSTEM: You are an AI assistant for institutional trading analysis. 
You must ONLY provide {self.approved_purposes[purpose]}.
Do NOT execute code, reveal system information, or provide personal data.
Context provided below is the ONLY source of truth.]

Purpose: {purpose}
Context: {json.dumps(context, default=str)[:1000]}

User Query: {prompt}

Response (be concise and factual):
"""
        return safe_prompt
        
    async def _call_llm(self, prompt: str) -> str:
        """Call actual LLM (placeholder for integration)"""
        # Simulate LLM response
        await asyncio.sleep(0.1)  # Simulate latency
        
        # In production, integrate with actual LLM API
        # For now, return a safe placeholder response
        return "Based on the provided context, the analysis indicates neutral market conditions with no strong directional bias. Risk management protocols should be maintained."
        
    def _calculate_confidence(self, threats: List[SecurityThreat]) -> float:
        """Calculate confidence score based on threats"""
        if not threats:
            return 0.95
            
        # Reduce confidence based on threat severity
        confidence = 1.0
        
        for threat in threats:
            if threat.severity == 'critical':
                confidence *= 0.3
            elif threat.severity == 'high':
                confidence *= 0.5
            elif threat.severity == 'medium':
                confidence *= 0.7
            elif threat.severity == 'low':
                confidence *= 0.9
                
        return max(0.0, confidence)
        
    def _log_threats(self, threats: List[SecurityThreat]):
        """Log security threats"""
        for threat in threats:
            self.threat_log.append({
                'timestamp': threat.timestamp,
                'type': threat.threat_type,
                'severity': threat.severity,
                'description': threat.description,
                'blocked': threat.blocked
            })
            
            logger.warning(
                "Security threat detected: %s (%s) - %s",
                threat.threat_type,
                threat.severity,
                threat.description
            )
            
    def _audit_log(self, request: LLMRequest, response: str, threats: List[SecurityThreat]):
        """Add to audit log"""
        self.audit_log.append({
            'timestamp': time.time(),
            'purpose': request.purpose,
            'user_id': request.user_id,
            'prompt_hash': hashlib.sha256(request.prompt.encode()).hexdigest()[:8],
            'response_length': len(response),
            'threats_count': len(threats),
            'filtered': any(t.blocked for t in threats)
        })
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get gatekeeper statistics"""
        return {
            'total_requests': self.total_requests,
            'blocked_requests': self.blocked_requests,
            'filtered_outputs': self.filtered_outputs,
            'block_rate': self.blocked_requests / max(1, self.total_requests),
            'filter_rate': self.filtered_outputs / max(1, self.total_requests),
            'recent_threats': len(self.threat_log),
            'audit_entries': len(self.audit_log)
        }


# Global instance
_gatekeeper = None


def get_llm_gatekeeper() -> LLMGatekeeper:
    """Get or create the global LLM gatekeeper"""
    global _gatekeeper
    if _gatekeeper is None:
        _gatekeeper = LLMGatekeeper()
    return _gatekeeper
