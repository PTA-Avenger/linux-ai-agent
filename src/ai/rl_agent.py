"""
Reinforcement Learning agent placeholder for Linux AI Agent.
"""

import json
import os
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import get_logger, log_operation


logger = get_logger("rl_agent")


class RLAgent:
    """
    Reinforcement Learning agent placeholder.
    This is a simplified implementation that can be extended with actual RL algorithms.
    """
    
    def __init__(self, model_path: str = "rl_model.json"):
        self.model_path = Path(model_path)
        self.actions_taken = []
        self.rewards = []
        self.state_history = []
        
        # Simple Q-table representation (placeholder)
        self.q_table = {}
        
        # Load existing model if available
        self._load_model()
        
        # Define action space
        self.action_space = [
            "scan_file",
            "quarantine_file",
            "delete_file",
            "ignore_file",
            "update_database",
            "monitor_system",
        ]
        
        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # Exploration rate
        
        logger.info("RL Agent initialized")
    
    def _load_model(self):
        """Load the RL model from file."""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'r') as f:
                    data = json.load(f)
                    self.q_table = data.get("q_table", {})
                    self.actions_taken = data.get("actions_taken", [])
                    self.rewards = data.get("rewards", [])
                    logger.info("RL model loaded successfully")
            else:
                logger.info("No existing RL model found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading RL model: {e}")
            self.q_table = {}
    
    def _save_model(self):
        """Save the RL model to file."""
        try:
            data = {
                "q_table": self.q_table,
                "actions_taken": self.actions_taken[-1000:],  # Keep last 1000 actions
                "rewards": self.rewards[-1000:],  # Keep last 1000 rewards
                "last_updated": time.time()
            }
            
            with open(self.model_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info("RL model saved successfully")
        except Exception as e:
            logger.error(f"Error saving RL model: {e}")
    
    def get_state(self, context: Dict[str, Any]) -> str:
        """
        Convert context to a state representation.
        
        Args:
            context: Current context information
        
        Returns:
            State string representation
        """
        # Simple state representation based on context
        state_features = []
        
        # File-related features
        if "file_size" in context:
            size = context["file_size"]
            if size < 1024:
                state_features.append("small_file")
            elif size < 1024 * 1024:
                state_features.append("medium_file")
            else:
                state_features.append("large_file")
        
        # Scan results features
        if "scan_results" in context:
            results = context["scan_results"]
            if results.get("infected", False):
                state_features.append("infected")
            elif results.get("suspicious", False):
                state_features.append("suspicious")
            else:
                state_features.append("clean")
        
        # Entropy features
        if "entropy" in context:
            entropy = context["entropy"]
            if entropy > 7.5:
                state_features.append("high_entropy")
            elif entropy > 5.0:
                state_features.append("medium_entropy")
            else:
                state_features.append("low_entropy")
        
        # System load features
        if "system_load" in context:
            load = context["system_load"]
            if load > 80:
                state_features.append("high_load")
            elif load > 50:
                state_features.append("medium_load")
            else:
                state_features.append("low_load")
        
        # Create state string
        state = "_".join(sorted(state_features)) if state_features else "default"
        return state
    
    def select_action(self, state: str, available_actions: Optional[List[str]] = None) -> str:
        """
        Select an action based on current state using epsilon-greedy policy.
        
        Args:
            state: Current state string
            available_actions: List of available actions (defaults to all actions)
        
        Returns:
            Selected action
        """
        if available_actions is None:
            available_actions = self.action_space
        
        # Initialize Q-values for new state
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.action_space}
        
        # Epsilon-greedy action selection
        if len(self.actions_taken) < 10 or (len(self.actions_taken) % 100 == 0):
            # Exploration: random action
            import random
            action = random.choice(available_actions)
            exploration = True
        else:
            # Exploitation: best known action
            state_q_values = self.q_table[state]
            available_q_values = {
                action: state_q_values.get(action, 0.0) 
                for action in available_actions
            }
            action = max(available_q_values, key=available_q_values.get)
            exploration = False
        
        log_operation(logger, "SELECT_ACTION", {
            "state": state,
            "action": action,
            "exploration": exploration,
            "available_actions": len(available_actions)
        })
        
        return action
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """
        Update Q-value using Q-learning algorithm.
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: New state after action
        """
        # Initialize states if not exist
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.action_space}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.action_space}
        
        # Q-learning update
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
        
        log_operation(logger, "UPDATE_Q_VALUE", {
            "state": state,
            "action": action,
            "reward": reward,
            "old_q": current_q,
            "new_q": new_q
        })
    
    def record_experience(self, state: str, action: str, reward: float, next_state: str):
        """
        Record an experience and update the model.
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: New state after action
        """
        # Record the experience
        experience = {
            "timestamp": time.time(),
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state
        }
        
        self.actions_taken.append(experience)
        self.rewards.append(reward)
        
        # Update Q-value
        self.update_q_value(state, action, reward, next_state)
        
        # Save model periodically
        if len(self.actions_taken) % 10 == 0:
            self._save_model()
        
        log_operation(logger, "RECORD_EXPERIENCE", {
            "state": state,
            "action": action,
            "reward": reward,
            "total_experiences": len(self.actions_taken)
        })
    
    def calculate_reward(self, action: str, results: Dict[str, Any]) -> float:
        """
        Calculate reward based on action results.
        
        Args:
            action: Action that was taken
            results: Results of the action
        
        Returns:
            Reward value
        """
        reward = 0.0
        
        if action == "scan_file":
            if results.get("status") == "success":
                reward += 1.0
                if results.get("infected", False):
                    reward += 5.0  # High reward for detecting infection
                elif results.get("suspicious", False):
                    reward += 2.0  # Medium reward for detecting suspicious file
            else:
                reward -= 1.0  # Penalty for failed scan
        
        elif action == "quarantine_file":
            if results.get("status") == "success":
                reward += 3.0  # Good reward for successful quarantine
            else:
                reward -= 2.0  # Penalty for failed quarantine
        
        elif action == "delete_file":
            if results.get("status") == "success":
                reward += 2.0  # Reward for successful deletion
            else:
                reward -= 2.0  # Penalty for failed deletion
        
        elif action == "ignore_file":
            # Reward depends on whether file was actually safe
            if not results.get("infected", False) and not results.get("suspicious", False):
                reward += 0.5  # Small reward for correctly ignoring safe file
            else:
                reward -= 3.0  # Penalty for ignoring dangerous file
        
        elif action == "update_database":
            if results.get("status") == "success":
                reward += 1.0
            else:
                reward -= 0.5
        
        elif action == "monitor_system":
            reward += 0.1  # Small positive reward for monitoring
        
        return reward
    
    def get_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get action recommendations based on current context.
        
        Args:
            context: Current context information
        
        Returns:
            List of recommended actions with confidence scores
        """
        state = self.get_state(context)
        
        if state not in self.q_table:
            # No experience with this state, return default recommendations
            return [
                {"action": "scan_file", "confidence": 0.5, "reason": "Default action for unknown state"},
                {"action": "monitor_system", "confidence": 0.3, "reason": "Safe monitoring action"}
            ]
        
        # Get Q-values for current state
        q_values = self.q_table[state]
        
        # Sort actions by Q-value
        sorted_actions = sorted(q_values.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for action, q_value in sorted_actions[:3]:  # Top 3 actions
            confidence = max(0.0, min(1.0, (q_value + 10) / 20))  # Normalize to 0-1
            
            recommendations.append({
                "action": action,
                "confidence": confidence,
                "q_value": q_value,
                "reason": f"Learned from {len(self.actions_taken)} experiences"
            })
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        Returns:
            Dictionary with agent statistics
        """
        stats = {
            "total_experiences": len(self.actions_taken),
            "total_states": len(self.q_table),
            "average_reward": sum(self.rewards) / len(self.rewards) if self.rewards else 0.0,
            "recent_average_reward": sum(self.rewards[-100:]) / min(100, len(self.rewards)) if self.rewards else 0.0,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon
        }
        
        # Action frequency
        if self.actions_taken:
            action_counts = {}
            for exp in self.actions_taken:
                action = exp["action"]
                action_counts[action] = action_counts.get(action, 0) + 1
            
            stats["action_frequencies"] = action_counts
        
        return stats
    
    def reset_model(self):
        """Reset the RL model to start fresh."""
        self.q_table = {}
        self.actions_taken = []
        self.rewards = []
        self.state_history = []
        
        # Remove model file
        if self.model_path.exists():
            self.model_path.unlink()
        
        logger.info("RL model reset successfully")
        
        log_operation(logger, "RESET_MODEL", {
            "timestamp": time.time()
        })
