"""
Cross-Agent Reasoning Pipeline
Coordinates multiple AI agents for collective decision making and consensus building
"""

import logging
import time
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Message passed between agents"""
    sender: str
    receiver: str
    content: Dict[str, Any]
    priority: float = 0.5
    timestamp: float = field(default_factory=time.time)
    requires_response: bool = False
    correlation_id: Optional[str] = None


@dataclass
class AgentOpinion:
    """Individual agent's opinion on a decision"""
    agent_name: str
    decision: str  # 'buy', 'sell', 'hold'
    confidence: float
    reasoning: str
    supporting_data: Dict[str, Any]
    dissenting_points: List[str] = field(default_factory=list)


@dataclass
class ConsensusDecision:
    """Final consensus decision from all agents"""
    decision: str
    confidence: float
    consensus_level: float  # 0-1, how much agents agree
    participating_agents: List[str]
    opinions: List[AgentOpinion]
    reasoning: str
    execution_params: Dict[str, Any]


class ReasoningAgent:
    """Base class for reasoning agents"""
    
    def __init__(self, name: str, specialization: str):
        self.name = name
        self.specialization = specialization
        self.message_queue = asyncio.Queue()
        self.knowledge_base = {}
        self.performance_history = deque(maxlen=100)
        self.trust_scores = {}  # Trust in other agents
        
    async def process_message(self, message: AgentMessage) -> Optional[Dict]:
        """Process incoming message and optionally respond"""
        
        # Update knowledge from message
        self.knowledge_base[message.sender] = message.content
        
        if message.requires_response:
            # Generate response based on specialization
            response = await self._generate_response(message)
            return response
            
        return None
        
    async def _generate_response(self, message: AgentMessage) -> Dict:
        """Generate response based on agent's specialization"""
        # Override in subclasses
        return {"status": "acknowledged"}
        
    async def form_opinion(self, market_data: Dict, signal_data: Dict) -> AgentOpinion:
        """Form opinion on trading decision"""
        # Override in subclasses
        return AgentOpinion(
            agent_name=self.name,
            decision="hold",
            confidence=0.5,
            reasoning="Base agent default opinion",
            supporting_data={}
        )
        
    def update_trust(self, agent: str, outcome: float):
        """Update trust score for another agent based on outcome"""
        if agent not in self.trust_scores:
            self.trust_scores[agent] = 0.5
            
        # Exponential moving average update
        alpha = 0.1
        self.trust_scores[agent] = (1 - alpha) * self.trust_scores[agent] + alpha * outcome


class TechnicalAnalysisAgent(ReasoningAgent):
    """Agent specialized in technical analysis"""
    
    def __init__(self):
        super().__init__("TechnicalAgent", "technical_analysis")
        
    async def form_opinion(self, market_data: Dict, signal_data: Dict) -> AgentOpinion:
        """Form opinion based on technical indicators"""
        
        # Analyze technical signals
        rsi = market_data.get('rsi', 50)
        macd = market_data.get('macd', 0)
        bb_position = market_data.get('bb_position', 0.5)
        volume_trend = market_data.get('volume_trend', 0)
        
        # Decision logic
        bullish_signals = 0
        bearish_signals = 0
        
        if rsi < 30:
            bullish_signals += 1
        elif rsi > 70:
            bearish_signals += 1
            
        if macd > 0:
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        if bb_position < 0.2:
            bullish_signals += 1
        elif bb_position > 0.8:
            bearish_signals += 1
            
        if volume_trend > 0:
            bullish_signals += 0.5
            
        # Form decision
        if bullish_signals > bearish_signals + 1:
            decision = "buy"
            confidence = min(0.9, bullish_signals / 4)
        elif bearish_signals > bullish_signals + 1:
            decision = "sell"
            confidence = min(0.9, bearish_signals / 4)
        else:
            decision = "hold"
            confidence = 0.4
            
        reasoning = f"Technical signals: {bullish_signals} bullish, {bearish_signals} bearish. "
        reasoning += f"RSI={rsi:.0f}, MACD={macd:.3f}, BB={bb_position:.2f}"
        
        return AgentOpinion(
            agent_name=self.name,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data={
                'rsi': rsi,
                'macd': macd,
                'bb_position': bb_position,
                'volume_trend': volume_trend
            }
        )


class FundamentalAnalysisAgent(ReasoningAgent):
    """Agent specialized in fundamental/macro analysis"""
    
    def __init__(self):
        super().__init__("FundamentalAgent", "fundamental_analysis")
        
    async def form_opinion(self, market_data: Dict, signal_data: Dict) -> AgentOpinion:
        """Form opinion based on fundamental factors"""
        
        # Analyze fundamental data
        sentiment = signal_data.get('llm_macro', 0)
        news_sentiment = market_data.get('news_sentiment', 0)
        economic_outlook = market_data.get('economic_outlook', 0)
        
        # Combine fundamental signals
        fundamental_score = (sentiment + news_sentiment + economic_outlook) / 3
        
        # Form decision
        if fundamental_score > 0.2:
            decision = "buy"
            confidence = min(0.85, abs(fundamental_score))
        elif fundamental_score < -0.2:
            decision = "sell"
            confidence = min(0.85, abs(fundamental_score))
        else:
            decision = "hold"
            confidence = 0.3
            
        reasoning = f"Fundamental score: {fundamental_score:.2f}. "
        reasoning += f"Sentiment={sentiment:.2f}, News={news_sentiment:.2f}"
        
        return AgentOpinion(
            agent_name=self.name,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data={
                'sentiment': sentiment,
                'news_sentiment': news_sentiment,
                'economic_outlook': economic_outlook
            }
        )


class RiskManagementAgent(ReasoningAgent):
    """Agent specialized in risk assessment"""
    
    def __init__(self):
        super().__init__("RiskAgent", "risk_management")
        self.max_risk_tolerance = 0.02  # 2% max risk
        
    async def form_opinion(self, market_data: Dict, signal_data: Dict) -> AgentOpinion:
        """Form opinion based on risk factors"""
        
        # Analyze risk metrics
        volatility = market_data.get('volatility', 0.01)
        max_drawdown = market_data.get('max_drawdown', 0)
        var_95 = market_data.get('var_95', 0)
        correlation_risk = market_data.get('correlation_risk', 0)
        
        # Calculate risk score
        risk_score = (
            volatility * 0.3 +
            abs(max_drawdown) * 0.3 +
            var_95 * 0.2 +
            correlation_risk * 0.2
        )
        
        # Risk-based decision
        if risk_score > self.max_risk_tolerance:
            decision = "hold"  # Too risky
            confidence = 0.8
            dissenting = ["Risk exceeds tolerance"]
        else:
            # Defer to other signals but adjust confidence
            ai_consensus = np.mean(list(signal_data.values()))
            if ai_consensus > 0:
                decision = "buy"
            elif ai_consensus < 0:
                decision = "sell"
            else:
                decision = "hold"
                
            # Lower confidence based on risk
            confidence = max(0.3, 1.0 - risk_score / self.max_risk_tolerance)
            dissenting = []
            
        reasoning = f"Risk score: {risk_score:.3f} vs tolerance: {self.max_risk_tolerance}. "
        reasoning += f"Vol={volatility:.3f}, DD={max_drawdown:.1%}"
        
        return AgentOpinion(
            agent_name=self.name,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data={
                'risk_score': risk_score,
                'volatility': volatility,
                'max_drawdown': max_drawdown
            },
            dissenting_points=dissenting
        )


class PatternRecognitionAgent(ReasoningAgent):
    """Agent specialized in pattern recognition"""
    
    def __init__(self):
        super().__init__("PatternAgent", "pattern_recognition")
        
    async def form_opinion(self, market_data: Dict, signal_data: Dict) -> AgentOpinion:
        """Form opinion based on pattern recognition"""
        
        # Get pattern signals
        cnn_signal = signal_data.get('cnn', 0)
        visual_signal = signal_data.get('visual_ai', 0)
        pattern_strength = market_data.get('pattern_strength', 0)
        
        # Combine pattern signals
        pattern_score = (cnn_signal + visual_signal) / 2
        
        # Adjust by pattern strength
        if pattern_strength > 0:
            pattern_score *= (1 + pattern_strength)
            
        # Form decision
        if pattern_score > 0.3:
            decision = "buy"
            confidence = min(0.9, abs(pattern_score))
        elif pattern_score < -0.3:
            decision = "sell"
            confidence = min(0.9, abs(pattern_score))
        else:
            decision = "hold"
            confidence = 0.4
            
        reasoning = f"Pattern score: {pattern_score:.2f}. "
        reasoning += f"CNN={cnn_signal:.2f}, Visual={visual_signal:.2f}, Strength={pattern_strength:.2f}"
        
        return AgentOpinion(
            agent_name=self.name,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data={
                'cnn_signal': cnn_signal,
                'visual_signal': visual_signal,
                'pattern_strength': pattern_strength
            }
        )


class ReinforcementLearningAgent(ReasoningAgent):
    """Agent using RL for decision making"""
    
    def __init__(self):
        super().__init__("RLAgent", "reinforcement_learning")
        
    async def form_opinion(self, market_data: Dict, signal_data: Dict) -> AgentOpinion:
        """Form opinion based on RL policy"""
        
        # Get RL signals
        ppo_signal = signal_data.get('ppo', 0)
        q_value = market_data.get('q_value', 0)
        expected_reward = market_data.get('expected_reward', 0)
        
        # RL decision
        rl_score = ppo_signal * 0.6 + q_value * 0.2 + expected_reward * 0.2
        
        if rl_score > 0.2:
            decision = "buy"
            confidence = min(0.85, abs(rl_score))
        elif rl_score < -0.2:
            decision = "sell"
            confidence = min(0.85, abs(rl_score))
        else:
            decision = "hold"
            confidence = 0.5
            
        reasoning = f"RL score: {rl_score:.2f}. "
        reasoning += f"PPO={ppo_signal:.2f}, Q={q_value:.2f}, Reward={expected_reward:.2f}"
        
        return AgentOpinion(
            agent_name=self.name,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data={
                'ppo_signal': ppo_signal,
                'q_value': q_value,
                'expected_reward': expected_reward
            }
        )


class CrossAgentReasoning:
    """Main cross-agent reasoning coordinator"""
    
    def __init__(self):
        # Initialize agents
        self.agents = {
            'technical': TechnicalAnalysisAgent(),
            'fundamental': FundamentalAnalysisAgent(),
            'risk': RiskManagementAgent(),
            'pattern': PatternRecognitionAgent(),
            'rl': ReinforcementLearningAgent()
        }
        
        # Communication channels
        self.message_bus = asyncio.Queue()
        self.consensus_history = deque(maxlen=100)
        
        # Consensus parameters
        self.min_agents_for_consensus = 3
        self.confidence_threshold = 0.6
        self.debate_rounds = 2
        
    async def broadcast_message(self, sender: str, content: Dict, 
                               priority: float = 0.5):
        """Broadcast message to all agents"""
        
        for agent_name in self.agents:
            if agent_name != sender:
                message = AgentMessage(
                    sender=sender,
                    receiver=agent_name,
                    content=content,
                    priority=priority
                )
                await self.agents[agent_name].message_queue.put(message)
                
    async def gather_opinions(self, market_data: Dict, 
                            signal_data: Dict) -> List[AgentOpinion]:
        """Gather opinions from all agents"""
        
        opinions = []
        tasks = []
        
        for agent_name, agent in self.agents.items():
            task = agent.form_opinion(market_data, signal_data)
            tasks.append(task)
            
        # Gather all opinions in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, AgentOpinion):
                opinions.append(result)
            else:
                logger.error(f"Agent opinion error: {result}")
                
        return opinions
        
    async def debate_phase(self, opinions: List[AgentOpinion], 
                         rounds: int = 2) -> List[AgentOpinion]:
        """Allow agents to debate and potentially change opinions"""
        
        for round_num in range(rounds):
            # Share opinions among agents
            for opinion in opinions:
                await self.broadcast_message(
                    opinion.agent_name,
                    {
                        'decision': opinion.decision,
                        'confidence': opinion.confidence,
                        'reasoning': opinion.reasoning
                    },
                    priority=opinion.confidence
                )
                
            # Allow agents to process messages
            await asyncio.sleep(0.01)  # Brief pause for message processing
            
            # Agents can adjust opinions based on others' input
            # (In a full implementation, agents would process their queues here)
            
        return opinions
        
    def calculate_consensus(self, opinions: List[AgentOpinion]) -> ConsensusDecision:
        """Calculate consensus from agent opinions"""
        
        if not opinions:
            return ConsensusDecision(
                decision="hold",
                confidence=0.0,
                consensus_level=0.0,
                participating_agents=[],
                opinions=[],
                reasoning="No agent opinions available",
                execution_params={}
            )
            
        # Count decisions weighted by confidence
        decision_weights = {'buy': 0, 'sell': 0, 'hold': 0}
        total_confidence = 0
        
        for opinion in opinions:
            # Apply agent trust scores if available
            trust = 1.0  # Default trust
            
            # Weight by confidence and trust
            weight = opinion.confidence * trust
            decision_weights[opinion.decision] += weight
            total_confidence += opinion.confidence
            
        # Normalize weights
        if total_confidence > 0:
            for decision in decision_weights:
                decision_weights[decision] /= total_confidence
                
        # Determine consensus decision
        consensus_decision = max(decision_weights, key=decision_weights.get)
        consensus_confidence = decision_weights[consensus_decision]
        
        # Calculate consensus level (how much agents agree)
        max_weight = max(decision_weights.values())
        consensus_level = max_weight  # 1.0 means full agreement
        
        # Check for strong dissent
        dissenting_agents = [
            op.agent_name for op in opinions 
            if op.decision != consensus_decision and op.confidence > 0.7
        ]
        
        # Adjust confidence if there's strong dissent
        if dissenting_agents:
            consensus_confidence *= 0.8
            
        # Build reasoning
        reasoning = f"Consensus: {consensus_decision} ({consensus_confidence:.1%}). "
        reasoning += f"Agreement level: {consensus_level:.1%}. "
        
        if dissenting_agents:
            reasoning += f"Dissent from: {', '.join(dissenting_agents)}. "
            
        # Add top supporting reasons
        supporting = [op for op in opinions if op.decision == consensus_decision]
        if supporting:
            top_reason = max(supporting, key=lambda x: x.confidence)
            reasoning += f"Primary: {top_reason.reasoning}"
            
        # Determine execution parameters
        execution_params = self._determine_execution_params(
            consensus_decision, 
            consensus_confidence,
            opinions
        )
        
        return ConsensusDecision(
            decision=consensus_decision,
            confidence=consensus_confidence,
            consensus_level=consensus_level,
            participating_agents=[op.agent_name for op in opinions],
            opinions=opinions,
            reasoning=reasoning,
            execution_params=execution_params
        )
        
    def _determine_execution_params(self, decision: str, confidence: float,
                                   opinions: List[AgentOpinion]) -> Dict:
        """Determine execution parameters based on consensus"""
        
        # Get risk agent's opinion
        risk_opinion = next((op for op in opinions if op.agent_name == "RiskAgent"), None)
        
        # Base position size on confidence and risk
        if risk_opinion:
            risk_factor = 1.0 - risk_opinion.supporting_data.get('risk_score', 0) / 0.02
            position_size = confidence * risk_factor
        else:
            position_size = confidence * 0.8
            
        # Determine time horizon
        if confidence > 0.8:
            time_horizon = "swing"  # High confidence, longer hold
        elif confidence > 0.6:
            time_horizon = "intraday"
        else:
            time_horizon = "scalp"  # Low confidence, quick exit
            
        return {
            'position_size': min(1.0, max(0.1, position_size)),
            'time_horizon': time_horizon,
            'stop_loss_multiplier': 1.5 if confidence < 0.6 else 1.0,
            'take_profit_multiplier': 2.0 if confidence > 0.7 else 1.5,
            'trailing_stop': confidence > 0.7
        }
        
    async def reason(self, market_data: Dict, signal_data: Dict) -> ConsensusDecision:
        """
        Main reasoning pipeline
        Coordinates all agents to reach a consensus decision
        """
        
        try:
            # Phase 1: Gather initial opinions
            opinions = await self.gather_opinions(market_data, signal_data)
            
            if len(opinions) < self.min_agents_for_consensus:
                logger.warning(f"Insufficient agents: {len(opinions)} < {self.min_agents_for_consensus}")
                
            # Phase 2: Debate (agents can adjust opinions)
            if self.debate_rounds > 0:
                opinions = await self.debate_phase(opinions, self.debate_rounds)
                
            # Phase 3: Calculate consensus
            consensus = self.calculate_consensus(opinions)
            
            # Store in history
            self.consensus_history.append({
                'timestamp': time.time(),
                'consensus': consensus,
                'market_snapshot': market_data
            })
            
            # Log decision
            logger.info(
                "Cross-Agent Consensus: %s (%.1f%%) | Level: %.1f%% | Agents: %d",
                consensus.decision,
                consensus.confidence * 100,
                consensus.consensus_level * 100,
                len(consensus.participating_agents)
            )
            
            return consensus
            
        except Exception as e:
            logger.error(f"Cross-agent reasoning failed: {e}")
            
            # Return safe default
            return ConsensusDecision(
                decision="hold",
                confidence=0.0,
                consensus_level=0.0,
                participating_agents=[],
                opinions=[],
                reasoning=f"Reasoning failed: {str(e)}",
                execution_params={'position_size': 0}
            )
            
    def update_agent_performance(self, agent_name: str, outcome: float):
        """Update agent performance based on trade outcome"""
        
        if agent_name in self.agents:
            agent = self.agents[agent_name]
            agent.performance_history.append(outcome)
            
            # Update trust scores between agents
            for other_agent in self.agents.values():
                if other_agent.name != agent_name:
                    other_agent.update_trust(agent_name, outcome)
                    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics for all agents"""
        
        stats = {}
        
        for name, agent in self.agents.items():
            perf_history = list(agent.performance_history)
            
            stats[name] = {
                'specialization': agent.specialization,
                'total_decisions': len(perf_history),
                'avg_performance': np.mean(perf_history) if perf_history else 0,
                'trust_scores': agent.trust_scores,
                'knowledge_sources': len(agent.knowledge_base)
            }
            
        return stats


# Global instance
_cross_agent_reasoning = None


def get_cross_agent_reasoning() -> CrossAgentReasoning:
    """Get or create the global cross-agent reasoning system"""
    global _cross_agent_reasoning
    if _cross_agent_reasoning is None:
        _cross_agent_reasoning = CrossAgentReasoning()
    return _cross_agent_reasoning
