"""
PPO Trainer - Institutional-grade Reinforcement Learning for Forex Trading
Produces stable-baselines3 PPO agent with proper reward engineering and validation
"""

import os
import pathlib
import argparse
import json
import numpy as np
import random
import time
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import torch

PROJECT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.getenv("ARIA_DATA_ROOT", PROJECT / "data"))
DEVICE = torch.device("cpu")

# CPU optimization for T470
torch.set_num_threads(2)


class ForexTradingEnv(gym.Env):
    """Institutional-grade Forex trading environment for PPO"""
    
    def __init__(self, data: np.ndarray, window_size: int = 50, initial_balance: float = 10000.0):
        super().__init__()
        
        self.data = data  # OHLCV + features
        self.window_size = window_size
        self.initial_balance = initial_balance
        
        # Action space: [hold, buy, sell] with position sizing
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: OHLC + technical indicators (normalized)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Random start position (leave room for window)
        max_start = len(self.data) - self.window_size - 100
        self.current_step = random.randint(self.window_size, max_start)
        
        self.balance = self.initial_balance
        self.position = 0.0  # Current position size (-1 to 1)
        self.entry_price = 0.0
        self.total_reward = 0.0
        self.trades_count = 0
        self.winning_trades = 0
        
        # Performance tracking
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current market observation (4 features)"""
        if self.current_step < self.window_size:
            return np.zeros(4, dtype=np.float32)
        
        # Get recent price data
        recent_data = self.data[self.current_step - self.window_size:self.current_step]
        closes = recent_data[:, 3]  # Close prices
        
        # Calculate 4 key features for PPO
        if len(closes) < 2:
            return np.zeros(4, dtype=np.float32)
        
        # 1. Price momentum (normalized)
        price_change = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0.0
        
        # 2. Volatility (rolling std)
        volatility = np.std(closes[-20:]) / np.mean(closes[-20:]) if len(closes) >= 20 else 0.0
        
        # 3. Trend strength (slope)
        x = np.arange(len(closes))
        if len(x) > 1:
            trend = np.polyfit(x, closes, 1)[0] / np.mean(closes)
        else:
            trend = 0.0
        
        # 4. Current position (for position awareness)
        position_feature = self.position
        
        obs = np.array([
            np.tanh(price_change * 10),  # Normalized price change
            np.tanh(volatility * 100),   # Normalized volatility
            np.tanh(trend * 1000),       # Normalized trend
            position_feature             # Current position
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action: np.ndarray):
        """Execute trading action and return reward"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0.0, True, True, {}
        
        # Extract action (position target)
        target_position = np.clip(action[0], -1.0, 1.0)
        
        # Get current and next prices
        current_price = self.data[self.current_step, 3]  # Close price
        next_price = self.data[self.current_step + 1, 3]
        
        # Calculate position change
        position_change = target_position - self.position
        
        # Execute trade if significant position change
        if abs(position_change) > 0.1:  # Minimum position change threshold
            self.trades_count += 1
            
            # Calculate transaction costs (spread + slippage)
            transaction_cost = abs(position_change) * 0.0002  # 2 pips per trade
            
            # Update position
            old_position = self.position
            self.position = target_position
            self.entry_price = current_price
            
            # Apply transaction cost
            self.balance -= transaction_cost * self.initial_balance
        
        # Calculate P&L from price movement
        price_return = (next_price - current_price) / current_price if current_price > 0 else 0.0
        position_pnl = self.position * price_return * self.initial_balance
        
        # Update balance
        old_balance = self.balance
        self.balance += position_pnl
        
        # Track performance metrics
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        current_drawdown = (self.peak_balance - self.balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Calculate reward (institutional-grade reward engineering)
        reward = self._calculate_reward(position_pnl, old_balance)
        self.total_reward += reward
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = (
            self.current_step >= len(self.data) - 1 or
            self.balance <= self.initial_balance * 0.5 or  # 50% drawdown limit
            self.max_drawdown > 0.3  # 30% max drawdown
        )
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'trades': self.trades_count,
            'max_drawdown': self.max_drawdown,
            'total_return': (self.balance - self.initial_balance) / self.initial_balance
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _calculate_reward(self, pnl: float, old_balance: float) -> float:
        """Institutional-grade reward function"""
        # Base reward from P&L
        pnl_reward = pnl / self.initial_balance * 100  # Scale to percentage
        
        # Risk-adjusted reward (Sharpe-like)
        if self.max_drawdown > 0:
            risk_penalty = -self.max_drawdown * 10  # Penalize drawdown
        else:
            risk_penalty = 0.0
        
        # Position holding penalty (encourage active trading)
        if abs(self.position) > 0.8:
            concentration_penalty = -0.1  # Penalize extreme positions
        else:
            concentration_penalty = 0.0
        
        # Combine rewards
        total_reward = pnl_reward + risk_penalty + concentration_penalty
        
        # Clip reward for stability
        return np.clip(total_reward, -10.0, 10.0)


def prepare_training_data(npz_path: pathlib.Path) -> np.ndarray:
    """Prepare data for PPO training"""
    data = np.load(npz_path, allow_pickle=False)
    
    # Extract OHLC data
    if 'open' in data and 'high' in data and 'low' in data and 'close' in data:
        ohlc = np.column_stack([
            data['open'],
            data['high'], 
            data['low'],
            data['close']
        ])
    else:
        # Fallback: create synthetic OHLC from close prices
        closes = data.get('close', data.get('r', np.random.randn(1000)))
        ohlc = np.column_stack([closes, closes, closes, closes])
    
    return ohlc.astype(np.float32)


def train_ppo(symbol: str, npz_path: pathlib.Path, config: Dict):
    """Main PPO training function"""
    print(f"Training PPO for {symbol}")
    
    # Set seeds for reproducibility
    seed = config.get('seed', 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Prepare data
    training_data = prepare_training_data(npz_path)
    print(f"Training data shape: {training_data.shape}")
    
    # Create environment
    def make_env():
        env = ForexTradingEnv(
            data=training_data,
            window_size=config.get('window_size', 50),
            initial_balance=config.get('initial_balance', 10000.0)
        )
        return Monitor(env)
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])
    
    # PPO configuration optimized for CPU
    ppo_config = {
        'learning_rate': config.get('learning_rate', 3e-4),
        'n_steps': config.get('n_steps', 2048),
        'batch_size': config.get('batch_size', 64),
        'n_epochs': config.get('n_epochs', 10),
        'gamma': config.get('gamma', 0.99),
        'gae_lambda': config.get('gae_lambda', 0.95),
        'clip_range': config.get('clip_range', 0.2),
        'ent_coef': config.get('ent_coef', 0.01),
        'vf_coef': config.get('vf_coef', 0.5),
        'max_grad_norm': config.get('max_grad_norm', 0.5),
        'device': 'cpu',
        'verbose': 1
    }
    
    # Create PPO agent
    model = PPO('MlpPolicy', env, **ppo_config)
    
    # Setup callbacks
    output_dir = DATA_ROOT / "models" / symbol
    output_dir.mkdir(parents=True, exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir),
        log_path=str(output_dir),
        eval_freq=config.get('eval_freq', 10000),
        deterministic=True,
        render=False
    )
    
    # Train the model
    print(f"Starting PPO training for {config['total_timesteps']} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=eval_callback,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f} seconds")
    
    # Save final model
    model_path = output_dir / "ppo_final.zip"
    model.save(str(model_path))
    
    # Save training metadata
    metadata = {
        'symbol': symbol,
        'training_time': training_time,
        'total_timesteps': config['total_timesteps'],
        'config': config,
        'model_path': str(model_path),
        'data_shape': training_data.shape,
        'environment': 'ForexTradingEnv'
    }
    
    metadata_path = output_dir / "ppo_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train PPO model for Forex trading")
    parser.add_argument("--symbol", required=True, help="Symbol to train")
    parser.add_argument("--npz", default=None, help="Path to NPZ file")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    args = parser.parse_args()
    
    # NPZ path
    if args.npz:
        npz_path = pathlib.Path(args.npz)
    else:
        npz_path = DATA_ROOT / "features_cache" / args.symbol / "train_m15.npz"
    
    if not npz_path.exists():
        print(f"Error: NPZ file not found at {npz_path}")
        return
    
    # Training configuration
    config = {
        'total_timesteps': args.timesteps,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'n_steps': 2048,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'eval_freq': 10000,
        'window_size': 50,
        'initial_balance': 10000.0,
        'seed': args.seed
    }
    
    # Train PPO
    metadata = train_ppo(args.symbol, npz_path, config)
    
    if metadata:
        print(f"\nPPO Training Complete!")
        print(f"Training time: {metadata['training_time']:.1f}s")
        print(f"Total timesteps: {metadata['total_timesteps']}")


if __name__ == "__main__":
    main()
