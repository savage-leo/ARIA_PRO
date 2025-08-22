"""
High-Frequency Arbitrage Detector
Real-time detection of arbitrage opportunities across multiple data feeds
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import deque
from datetime import datetime, timedelta
import json
from backend.core.performance_monitor import track_performance

logger = logging.getLogger(__name__)


class ArbitrageDetector:
    """
    High-frequency arbitrage detector for institutional trading
    Monitors multiple price feeds for exploitable inefficiencies
    """
    
    def __init__(self):
        self.active = True
        self.price_feeds = {}  # symbol -> feed -> price data
        self.latency_map = {}  # feed -> latency in ms
        self.arbitrage_history = deque(maxlen=1000)
        
        # Arbitrage parameters
        self.min_profit_pips = 2.0  # Minimum 2 pips profit after costs
        self.max_latency_ms = 50  # Max acceptable latency
        self.min_confidence = 0.8  # Minimum confidence to trade
        
        # Risk parameters
        self.max_position_size = 1.0  # Standard lots
        self.max_exposure = 5.0  # Max 5 lots total exposure
        self.current_exposure = 0.0
        
        # Statistical arbitrage params
        self.cointegration_window = 100
        self.zscore_threshold = 2.0
        
    @track_performance("ArbitrageDetector.add_price_feed")
    async def add_price_feed(self, feed_name: str, symbol: str, price_data: Dict):
        """Add price from a specific feed"""
        if symbol not in self.price_feeds:
            self.price_feeds[symbol] = {}
            
        timestamp = time.time()
        self.price_feeds[symbol][feed_name] = {
            'bid': price_data.get('bid'),
            'ask': price_data.get('ask'),
            'timestamp': timestamp,
            'volume': price_data.get('volume', 0)
        }
        
        # Update latency
        if 'server_time' in price_data:
            latency = (timestamp - price_data['server_time']) * 1000
            self.latency_map[feed_name] = latency
    
    @track_performance("ArbitrageDetector.detect_triangular_arbitrage")
    async def detect_triangular_arbitrage(self, symbols: List[str]) -> Optional[Dict]:
        """
        Detect triangular arbitrage opportunities
        e.g., EUR/USD -> USD/JPY -> EUR/JPY
        """
        if len(symbols) < 3:
            return None
            
        opportunities = []
        
        # Check common triangular pairs
        triangles = [
            ('EURUSD', 'USDJPY', 'EURJPY'),
            ('GBPUSD', 'USDJPY', 'GBPJPY'),
            ('EURUSD', 'GBPUSD', 'EURGBP'),
            ('AUDUSD', 'USDJPY', 'AUDJPY')
        ]
        
        for pair1, pair2, pair3 in triangles:
            if all(s in self.price_feeds for s in [pair1, pair2, pair3]):
                # Get best prices from all feeds
                p1 = self._get_best_price(pair1)
                p2 = self._get_best_price(pair2)
                p3 = self._get_best_price(pair3)
                
                if p1 and p2 and p3:
                    # Calculate synthetic price
                    if 'USD' in pair2:
                        synthetic = p1['bid'] * p2['bid']
                    else:
                        synthetic = p1['bid'] / p2['ask']
                    
                    # Compare with actual price
                    actual = p3['ask']
                    spread_pips = abs(synthetic - actual) * 10000
                    
                    if spread_pips > self.min_profit_pips:
                        opportunity = {
                            'type': 'triangular',
                            'pairs': [pair1, pair2, pair3],
                            'synthetic_price': synthetic,
                            'actual_price': actual,
                            'profit_pips': spread_pips,
                            'direction': 'buy' if synthetic < actual else 'sell',
                            'confidence': min(spread_pips / 10, 1.0),
                            'timestamp': datetime.now().isoformat()
                        }
                        opportunities.append(opportunity)
        
        if opportunities:
            best = max(opportunities, key=lambda x: x['profit_pips'])
            self.arbitrage_history.append(best)
            return best
            
        return None
    
    @track_performance("ArbitrageDetector.detect_latency_arbitrage")
    async def detect_latency_arbitrage(self, symbol: str) -> Optional[Dict]:
        """
        Detect latency arbitrage between different feeds
        Exploit slower feeds by trading on faster ones
        """
        if symbol not in self.price_feeds or len(self.price_feeds[symbol]) < 2:
            return None
            
        feeds = self.price_feeds[symbol]
        feed_names = list(feeds.keys())
        
        # Find fastest and slowest feeds
        latencies = [(name, self.latency_map.get(name, float('inf'))) 
                     for name in feed_names]
        latencies.sort(key=lambda x: x[1])
        
        if len(latencies) < 2:
            return None
            
        fast_feed = latencies[0][0]
        slow_feed = latencies[-1][0]
        
        # Check if latency difference is exploitable
        latency_diff = latencies[-1][1] - latencies[0][1]
        
        if latency_diff > 10:  # At least 10ms difference
            fast_price = feeds[fast_feed]
            slow_price = feeds[slow_feed]
            
            # Calculate price difference
            price_diff = fast_price['bid'] - slow_price['bid']
            spread_pips = abs(price_diff) * 10000
            
            if spread_pips > self.min_profit_pips:
                return {
                    'type': 'latency',
                    'symbol': symbol,
                    'fast_feed': fast_feed,
                    'slow_feed': slow_feed,
                    'latency_advantage': latency_diff,
                    'price_difference': price_diff,
                    'profit_pips': spread_pips,
                    'direction': 'buy' if price_diff > 0 else 'sell',
                    'confidence': min(latency_diff / 50, 1.0),
                    'timestamp': datetime.now().isoformat()
                }
        
        return None
    
    @track_performance("ArbitrageDetector.detect_statistical_arbitrage")
    async def detect_statistical_arbitrage(self, symbol_pair: Tuple[str, str]) -> Optional[Dict]:
        """
        Detect statistical arbitrage (pairs trading) opportunities
        Based on mean reversion of correlated pairs
        """
        symbol1, symbol2 = symbol_pair
        
        if symbol1 not in self.price_feeds or symbol2 not in self.price_feeds:
            return None
            
        # Get price histories (would need to maintain in production)
        # For now, use current prices to demonstrate
        price1 = self._get_best_price(symbol1)
        price2 = self._get_best_price(symbol2)
        
        if not price1 or not price2:
            return None
            
        # Calculate spread
        spread = price1['bid'] / price2['bid']
        
        # In production, would calculate z-score from historical spread
        # For demonstration, use synthetic z-score
        z_score = np.random.randn()  # Would be calculated from history
        
        if abs(z_score) > self.zscore_threshold:
            return {
                'type': 'statistical',
                'pair': [symbol1, symbol2],
                'spread': spread,
                'z_score': z_score,
                'direction': 'convergence' if z_score > 0 else 'divergence',
                'confidence': min(abs(z_score) / 3, 1.0),
                'expected_pips': abs(z_score) * 5,
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    @track_performance("ArbitrageDetector.detect_cross_broker_arbitrage")
    async def detect_cross_broker_arbitrage(self, symbol: str) -> Optional[Dict]:
        """
        Detect arbitrage between different brokers
        Requires multiple broker connections
        """
        if symbol not in self.price_feeds:
            return None
            
        feeds = self.price_feeds[symbol]
        if len(feeds) < 2:
            return None
            
        # Find best bid and ask across all feeds
        best_bid = max(feeds.items(), key=lambda x: x[1]['bid'])
        best_ask = min(feeds.items(), key=lambda x: x[1]['ask'])
        
        # Check if we can buy from one and sell to another
        if best_bid[0] != best_ask[0]:  # Different brokers
            spread = best_bid[1]['bid'] - best_ask[1]['ask']
            spread_pips = spread * 10000
            
            if spread_pips > self.min_profit_pips:
                return {
                    'type': 'cross_broker',
                    'symbol': symbol,
                    'buy_broker': best_ask[0],
                    'sell_broker': best_bid[0],
                    'buy_price': best_ask[1]['ask'],
                    'sell_price': best_bid[1]['bid'],
                    'profit_pips': spread_pips,
                    'confidence': min(spread_pips / 5, 1.0),
                    'timestamp': datetime.now().isoformat()
                }
        
        return None
    
    def _get_best_price(self, symbol: str) -> Optional[Dict]:
        """Get best bid/ask from all feeds for a symbol"""
        if symbol not in self.price_feeds:
            return None
            
        feeds = self.price_feeds[symbol]
        if not feeds:
            return None
            
        # Get most recent prices
        recent_feeds = {
            name: data for name, data in feeds.items()
            if time.time() - data['timestamp'] < 1.0  # Within 1 second
        }
        
        if not recent_feeds:
            return None
            
        # Find best bid and ask
        best_bid = max(recent_feeds.values(), key=lambda x: x['bid'])['bid']
        best_ask = min(recent_feeds.values(), key=lambda x: x['ask'])['ask']
        
        return {
            'bid': best_bid,
            'ask': best_ask,
            'timestamp': time.time()
        }
    
    @track_performance("ArbitrageDetector.execute_arbitrage")
    async def execute_arbitrage(self, opportunity: Dict) -> Dict:
        """
        Execute arbitrage trade
        Returns execution result
        """
        if opportunity['confidence'] < self.min_confidence:
            return {'status': 'skipped', 'reason': 'low_confidence'}
            
        if self.current_exposure >= self.max_exposure:
            return {'status': 'skipped', 'reason': 'max_exposure'}
            
        # Calculate position size based on opportunity type
        position_size = min(
            self.max_position_size,
            opportunity['confidence'] * self.max_position_size
        )
        
        # Log arbitrage execution
        logger.info(f"Executing arbitrage: {opportunity['type']} "
                   f"profit={opportunity.get('profit_pips', 0):.2f} pips")
        
        # In production, would execute actual trades
        # For now, simulate execution
        execution_result = {
            'status': 'executed',
            'type': opportunity['type'],
            'position_size': position_size,
            'expected_profit': opportunity.get('profit_pips', 0) * position_size,
            'timestamp': datetime.now().isoformat()
        }
        
        self.current_exposure += position_size
        
        return execution_result
    
    async def monitor_opportunities(self):
        """
        Main monitoring loop for arbitrage opportunities
        """
        while self.active:
            try:
                # Check all types of arbitrage
                tasks = []
                
                # Triangular arbitrage
                symbols = list(self.price_feeds.keys())
                tasks.append(self.detect_triangular_arbitrage(symbols))
                
                # Latency arbitrage for each symbol
                for symbol in symbols:
                    tasks.append(self.detect_latency_arbitrage(symbol))
                
                # Statistical arbitrage for major pairs
                pairs = [
                    ('EURUSD', 'GBPUSD'),
                    ('USDJPY', 'EURJPY'),
                    ('AUDUSD', 'NZDUSD')
                ]
                for pair in pairs:
                    tasks.append(self.detect_statistical_arbitrage(pair))
                
                # Cross-broker arbitrage
                for symbol in symbols:
                    tasks.append(self.detect_cross_broker_arbitrage(symbol))
                
                # Gather all results
                opportunities = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process valid opportunities
                valid_opportunities = [
                    opp for opp in opportunities 
                    if opp and not isinstance(opp, Exception)
                ]
                
                # Execute best opportunity
                if valid_opportunities:
                    best = max(valid_opportunities, 
                             key=lambda x: x.get('profit_pips', 0))
                    await self.execute_arbitrage(best)
                
                await asyncio.sleep(0.1)  # 100ms scan rate
                
            except Exception as e:
                logger.error(f"Arbitrage monitor error: {e}")
                await asyncio.sleep(1)
    
    def get_statistics(self) -> Dict:
        """Get arbitrage detection statistics"""
        if not self.arbitrage_history:
            return {}
            
        recent = list(self.arbitrage_history)[-100:]
        
        by_type = {}
        for arb in recent:
            arb_type = arb['type']
            if arb_type not in by_type:
                by_type[arb_type] = []
            by_type[arb_type].append(arb['profit_pips'])
        
        stats = {
            'total_opportunities': len(recent),
            'avg_profit_pips': np.mean([a.get('profit_pips', 0) for a in recent]),
            'by_type': {
                t: {
                    'count': len(pips),
                    'avg_pips': np.mean(pips),
                    'max_pips': max(pips)
                }
                for t, pips in by_type.items()
            },
            'current_exposure': self.current_exposure,
            'feed_latencies': self.latency_map
        }
        
        return stats


# Singleton instance
arbitrage_detector = ArbitrageDetector()
