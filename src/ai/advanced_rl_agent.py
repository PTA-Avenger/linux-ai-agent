"""
Advanced Reinforcement Learning Agent for Linux AI Agent.
Uses modern RL techniques including DQN, experience replay, and neural networks.
"""

import os
import sys
import json
import time
import random
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import get_logger, log_operation

# Import ML libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from sklearn.preprocessing import StandardScaler
    PYTORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available. Install with: pip install torch")
    PYTORCH_AVAILABLE = False

try:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    import gymnasium as gym
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    print("Stable Baselines3 not available. Install with: pip install stable-baselines3 gymnasium")
    STABLE_BASELINES_AVAILABLE = False
    # Create dummy gym for import compatibility
    class DummyGym:
        class Env:
            pass
        class spaces:
            @staticmethod
            def Discrete(n):
                return None
            @staticmethod 
            def Box(low, high, shape, dtype):
                return None
    gym = DummyGym()

logger = get_logger("advanced_rl_agent")


class SecurityEnvironment(gym.Env):
    """
    Custom Gymnasium environment for the security agent.
    """
    
    def __init__(self):
        super().__init__()
        
        # Define action space: 6 possible actions
        self.action_space = gym.spaces.Discrete(6)
        self.action_names = [
            "scan_file",
            "quarantine_file", 
            "delete_file",
            "ignore_file",
            "update_database",
            "monitor_system"
        ]
        
        # Define observation space: normalized features
        # Features: file_size, entropy, scan_result, system_load, time_of_day, file_age
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        self.current_state = None
        self.episode_length = 0
        self.max_episode_length = 100
        
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Generate initial state
        self.current_state = self._generate_random_state()
        self.episode_length = 0
        
        return self.current_state, {}
    
    def step(self, action):
        """Execute one step in the environment."""
        self.episode_length += 1
        
        # Calculate reward based on action and current state
        reward = self._calculate_reward(action, self.current_state)
        
        # Update state based on action
        self.current_state = self._update_state(action, self.current_state)
        
        # Check if episode is done
        done = (self.episode_length >= self.max_episode_length or 
                self._is_terminal_state(self.current_state))
        
        truncated = False
        info = {
            "action_name": self.action_names[action],
            "episode_length": self.episode_length
        }
        
        return self.current_state, reward, done, truncated, info
    
    def _generate_random_state(self) -> np.ndarray:
        """Generate a random state vector."""
        return np.random.rand(6).astype(np.float32)
    
    def _update_state(self, action: int, state: np.ndarray) -> np.ndarray:
        """Update state based on action taken."""
        new_state = state.copy()
        
        # Action effects on state
        if action == 0:  # scan_file
            new_state[2] = random.choice([0.0, 0.5, 1.0])  # scan_result
        elif action == 1:  # quarantine_file
            new_state[2] = 0.0  # file no longer a threat
        elif action == 2:  # delete_file
            new_state[0] = 0.0  # file size becomes 0
            new_state[2] = 0.0  # no longer a threat
        elif action == 4:  # update_database
            new_state[3] = max(0.0, new_state[3] - 0.1)  # reduce system load
        elif action == 5:  # monitor_system
            new_state[3] = min(1.0, new_state[3] + 0.05)  # slight increase in load
        
        # Add some noise
        new_state += np.random.normal(0, 0.01, size=new_state.shape)
        new_state = np.clip(new_state, 0.0, 1.0)
        
        return new_state
    
    def _calculate_reward(self, action: int, state: np.ndarray) -> float:
        """Calculate reward for action in given state."""
        file_size, entropy, scan_result, system_load, time_of_day, file_age = state
        
        reward = 0.0
        
        # Base reward for taking action
        reward += 0.1
        
        if action == 0:  # scan_file
            reward += 1.0
            if scan_result > 0.7:  # Found threat
                reward += 3.0
            elif scan_result > 0.3:  # Suspicious
                reward += 1.0
        
        elif action == 1:  # quarantine_file
            if scan_result > 0.5:  # Good to quarantine threats
                reward += 5.0
            else:  # Bad to quarantine clean files
                reward -= 3.0
        
        elif action == 2:  # delete_file
            if scan_result > 0.8:  # Good to delete confirmed threats
                reward += 4.0
            elif scan_result < 0.2:  # Bad to delete clean files
                reward -= 5.0
        
        elif action == 3:  # ignore_file
            if scan_result < 0.3:  # Good to ignore clean files
                reward += 1.0
            else:  # Bad to ignore threats
                reward -= 4.0
        
        elif action == 4:  # update_database
            reward += 0.5
            if system_load < 0.5:  # Better when system not busy
                reward += 0.5
        
        elif action == 5:  # monitor_system
            reward += 0.2
            if system_load > 0.7:  # More valuable when system busy
                reward += 0.3
        
        # Penalty for high system load
        reward -= system_load * 0.5
        
        return reward
    
    def _is_terminal_state(self, state: np.ndarray) -> bool:
        """Check if state is terminal."""
        # Terminal if system overloaded
        return state[3] > 0.95  # system_load > 95%


if PYTORCH_AVAILABLE:
    class DQNNetwork(nn.Module):
        """
        Deep Q-Network for the RL agent.
        """
        
        def __init__(self, state_size: int = 6, action_size: int = 6, hidden_size: int = 128):
            super(DQNNetwork, self).__init__()
            
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size, action_size)
            
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x
else:
    # Dummy DQNNetwork for compatibility
    class DQNNetwork:
        def __init__(self, *args, **kwargs):
            pass


class AdvancedRLAgent:
    """
    Advanced Reinforcement Learning Agent using modern techniques.
    """
    
    def __init__(self, model_path: str = "advanced_rl_model", use_stable_baselines: bool = True):
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        
        self.use_stable_baselines = use_stable_baselines and STABLE_BASELINES_AVAILABLE
        
        # Environment
        self.env = SecurityEnvironment()
        
        # Agent configuration
        self.state_size = 6
        self.action_size = 6
        self.learning_rate = 0.001
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.memory_size = 10000
        
        # Experience replay buffer
        self.memory = deque(maxlen=self.memory_size)
        
        # Neural networks (if PyTorch available)
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        
        # Stable Baselines model
        self.sb_model = None
        
        # Statistics
        self.training_episodes = 0
        self.total_rewards = []
        self.episode_lengths = []
        
        # Feature scaler
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
        self._initialize_models()
        
        logger.info(f"Advanced RL Agent initialized (Stable Baselines: {self.use_stable_baselines})")
    
    def _initialize_models(self):
        """Initialize the RL models."""
        if self.use_stable_baselines:
            self._initialize_stable_baselines()
        elif PYTORCH_AVAILABLE:
            self._initialize_pytorch_dqn()
        else:
            logger.warning("No RL libraries available, using simple Q-learning")
            self.q_table = {}
    
    def _initialize_stable_baselines(self):
        """Initialize Stable Baselines3 model."""
        try:
            model_file = self.model_path / "sb3_model.zip"
            
            if model_file.exists():
                self.sb_model = DQN.load(str(model_file), env=self.env)
                logger.info("Loaded existing Stable Baselines3 model")
            else:
                self.sb_model = DQN(
                    "MlpPolicy",
                    self.env,
                    learning_rate=self.learning_rate,
                    buffer_size=self.memory_size,
                    learning_starts=1000,
                    batch_size=self.batch_size,
                    target_update_interval=1000,
                    train_freq=4,
                    gradient_steps=1,
                    exploration_fraction=0.1,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.02,
                    verbose=1,
                    tensorboard_log=str(self.model_path / "tensorboard")
                )
                logger.info("Created new Stable Baselines3 DQN model")
                
        except Exception as e:
            logger.error(f"Error initializing Stable Baselines3: {e}")
            self.use_stable_baselines = False
            self._initialize_pytorch_dqn()
    
    def _initialize_pytorch_dqn(self):
        """Initialize PyTorch DQN."""
        try:
            self.q_network = DQNNetwork(self.state_size, self.action_size)
            self.target_network = DQNNetwork(self.state_size, self.action_size)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
            
            # Load existing model if available
            model_file = self.model_path / "pytorch_dqn.pth"
            if model_file.exists():
                checkpoint = torch.load(model_file)
                self.q_network.load_state_dict(checkpoint['q_network'])
                self.target_network.load_state_dict(checkpoint['target_network'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                logger.info("Loaded existing PyTorch DQN model")
            else:
                # Copy weights to target network
                self.target_network.load_state_dict(self.q_network.state_dict())
                logger.info("Created new PyTorch DQN model")
                
        except Exception as e:
            logger.error(f"Error initializing PyTorch DQN: {e}")
            self.q_network = None
    
    def get_state_vector(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Convert context to normalized state vector.
        
        Args:
            context: Current context information
            
        Returns:
            Normalized state vector
        """
        # Extract features from context
        file_size = context.get("file_size", 0)
        entropy = context.get("entropy", 0.0)
        scan_result = 1.0 if context.get("infected", False) else (0.5 if context.get("suspicious", False) else 0.0)
        system_load = context.get("system_load", 0.0) / 100.0  # Normalize to 0-1
        time_of_day = (time.time() % 86400) / 86400  # Time of day as fraction
        file_age = min(context.get("file_age", 0), 86400) / 86400  # File age in days, capped at 1 day
        
        # Normalize file size (log scale)
        if file_size > 0:
            file_size_norm = min(np.log10(file_size + 1) / 10, 1.0)  # Log scale, capped at 1
        else:
            file_size_norm = 0.0
        
        # Normalize entropy (0-8 scale)
        entropy_norm = min(entropy / 8.0, 1.0)
        
        state_vector = np.array([
            file_size_norm,
            entropy_norm,
            scan_result,
            system_load,
            time_of_day,
            file_age
        ], dtype=np.float32)
        
        # Apply scaling if fitted
        if self.scaler_fitted:
            state_vector = self.scaler.transform([state_vector])[0]
        
        return state_vector
    
    def select_action(self, context: Dict[str, Any], available_actions: Optional[List[str]] = None) -> str:
        """
        Select action using the trained model.
        
        Args:
            context: Current context
            available_actions: List of available actions
            
        Returns:
            Selected action name
        """
        state_vector = self.get_state_vector(context)
        
        if self.use_stable_baselines and self.sb_model:
            # Use Stable Baselines3 model
            action_idx, _ = self.sb_model.predict(state_vector, deterministic=False)
            action_name = self.env.action_names[action_idx]
            
        elif self.q_network is not None and PYTORCH_AVAILABLE:
            # Use PyTorch DQN
            if random.random() < self.epsilon:
                # Exploration
                action_idx = random.randint(0, self.action_size - 1)
            else:
                # Exploitation
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
                    q_values = self.q_network(state_tensor)
                    action_idx = q_values.argmax().item()
            
            action_name = self.env.action_names[action_idx]
            
        else:
            # Fallback to simple action selection
            action_name = self._select_action_simple(context)
        
        # Filter by available actions
        if available_actions and action_name not in available_actions:
            action_name = random.choice(available_actions)
        
        log_operation(logger, "SELECT_ACTION", {
            "action": action_name,
            "context_keys": list(context.keys()),
            "model_type": "stable_baselines" if self.use_stable_baselines else "pytorch_dqn"
        })
        
        return action_name
    
    def _select_action_simple(self, context: Dict[str, Any]) -> str:
        """Simple rule-based action selection as fallback."""
        if context.get("infected", False):
            return "quarantine_file"
        elif context.get("suspicious", False):
            return "scan_file"
        elif context.get("system_load", 0) > 80:
            return "monitor_system"
        else:
            return "scan_file"
    
    def train_episode(self, num_episodes: int = 100):
        """
        Train the agent for specified number of episodes.
        
        Args:
            num_episodes: Number of training episodes
        """
        if self.use_stable_baselines and self.sb_model:
            self._train_stable_baselines(num_episodes)
        elif self.q_network is not None:
            self._train_pytorch_dqn(num_episodes)
        else:
            logger.warning("No training method available")
    
    def _train_stable_baselines(self, num_episodes: int):
        """Train using Stable Baselines3."""
        try:
            # Calculate total timesteps (approximate)
            total_timesteps = num_episodes * self.env.max_episode_length
            
            logger.info(f"Starting Stable Baselines3 training for {total_timesteps} timesteps")
            
            self.sb_model.learn(
                total_timesteps=total_timesteps,
                callback=self._create_training_callback(),
                progress_bar=True
            )
            
            # Save model
            model_file = self.model_path / "sb3_model.zip"
            self.sb_model.save(str(model_file))
            
            logger.info("Stable Baselines3 training completed")
            
        except Exception as e:
            logger.error(f"Error in Stable Baselines3 training: {e}")
    
    def _train_pytorch_dqn(self, num_episodes: int):
        """Train using PyTorch DQN."""
        logger.info(f"Starting PyTorch DQN training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            episode_length = 0
            
            while True:
                # Select action
                if random.random() < self.epsilon:
                    action = random.randint(0, self.action_size - 1)
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        q_values = self.q_network(state_tensor)
                        action = q_values.argmax().item()
                
                # Take action
                next_state, reward, done, truncated, info = self.env.step(action)
                
                # Store experience
                self.memory.append((state, action, reward, next_state, done))
                
                # Update state and stats
                state = next_state
                total_reward += reward
                episode_length += 1
                
                # Train the network
                if len(self.memory) > self.batch_size:
                    self._replay_pytorch()
                
                if done or truncated:
                    break
            
            # Update target network periodically
            if episode % 100 == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Record statistics
            self.total_rewards.append(total_reward)
            self.episode_lengths.append(episode_length)
            self.training_episodes += 1
            
            if episode % 50 == 0:
                avg_reward = np.mean(self.total_rewards[-50:])
                logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        
        # Save model
        self._save_pytorch_model()
        logger.info("PyTorch DQN training completed")
    
    def _replay_pytorch(self):
        """Experience replay for PyTorch DQN."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _save_pytorch_model(self):
        """Save PyTorch model."""
        if self.q_network is None:
            return
        
        model_file = self.model_path / "pytorch_dqn.pth"
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_episodes': self.training_episodes
        }, model_file)
        
        logger.info("PyTorch model saved")
    
    def _create_training_callback(self):
        """Create training callback for Stable Baselines3."""
        class TrainingCallback(BaseCallback):
            def __init__(self, agent):
                super().__init__()
                self.agent = agent
            
            def _on_step(self) -> bool:
                return True
            
            def _on_rollout_end(self) -> None:
                # Log training progress
                if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                    recent_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer[-10:]]
                    if recent_rewards:
                        avg_reward = np.mean(recent_rewards)
                        logger.info(f"Recent average reward: {avg_reward:.2f}")
        
        return TrainingCallback(self)
    
    def record_experience(self, context: Dict[str, Any], action: str, reward: float, next_context: Dict[str, Any]):
        """
        Record experience for training.
        
        Args:
            context: Previous context
            action: Action taken
            reward: Reward received
            next_context: New context after action
        """
        if not self.use_stable_baselines and self.q_network is not None:
            # Convert to state vectors
            state = self.get_state_vector(context)
            next_state = self.get_state_vector(next_context)
            
            # Convert action name to index
            action_idx = self.env.action_names.index(action) if action in self.env.action_names else 0
            
            # Store in memory
            self.memory.append((state, action_idx, reward, next_state, False))
        
        log_operation(logger, "RECORD_EXPERIENCE", {
            "action": action,
            "reward": reward,
            "memory_size": len(self.memory) if hasattr(self, 'memory') else 0
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent training statistics."""
        stats = {
            "training_episodes": self.training_episodes,
            "model_type": "stable_baselines" if self.use_stable_baselines else "pytorch_dqn",
            "memory_size": len(self.memory) if hasattr(self, 'memory') else 0,
            "epsilon": getattr(self, 'epsilon', 0.0)
        }
        
        if self.total_rewards:
            stats.update({
                "average_reward": np.mean(self.total_rewards),
                "recent_average_reward": np.mean(self.total_rewards[-100:]) if len(self.total_rewards) > 0 else 0,
                "best_reward": max(self.total_rewards),
                "reward_trend": np.polyfit(range(len(self.total_rewards)), self.total_rewards, 1)[0] if len(self.total_rewards) > 1 else 0
            })
        
        if self.episode_lengths:
            stats.update({
                "average_episode_length": np.mean(self.episode_lengths),
                "recent_average_episode_length": np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) > 0 else 0
            })
        
        return stats
    
    def get_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get action recommendations with confidence scores.
        
        Args:
            context: Current context
            
        Returns:
            List of recommended actions
        """
        state_vector = self.get_state_vector(context)
        recommendations = []
        
        if self.use_stable_baselines and self.sb_model:
            # Get Q-values from Stable Baselines model
            try:
                # This is a bit hacky as SB3 doesn't directly expose Q-values
                action_probs = self.sb_model.policy.predict(state_vector, deterministic=False)[1]
                
                for i, prob in enumerate(action_probs):
                    recommendations.append({
                        "action": self.env.action_names[i],
                        "confidence": float(prob),
                        "reason": "Learned from RL training"
                    })
                    
            except Exception as e:
                logger.warning(f"Error getting SB3 recommendations: {e}")
                
        elif self.q_network is not None:
            # Get Q-values from PyTorch network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
                q_values = self.q_network(state_tensor).squeeze()
                
                # Convert to probabilities using softmax
                probs = F.softmax(q_values, dim=0)
                
                for i, (q_val, prob) in enumerate(zip(q_values, probs)):
                    recommendations.append({
                        "action": self.env.action_names[i],
                        "confidence": float(prob),
                        "q_value": float(q_val),
                        "reason": f"Q-value: {q_val:.3f}"
                    })
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        
        return recommendations[:3]  # Return top 3