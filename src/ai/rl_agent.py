"""
Reinforcement Learning agent for Linux AI Agent.
Enhanced with model persistence and improved decision making.
"""

import json
import os
import time
import random
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from collections import defaultdict, deque
import sys

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import get_logger, log_operation


logger = get_logger("rl_agent")


class RLAgent:
    """
    Enhanced Reinforcement Learning agent with Q-learning and experience replay.
    Now includes model persistence and improved decision recommendations.
    """
    
    def __init__(self, model_path: str = "rl_model.json", max_memory: int = 10000):
        self.model_path = Path(model_path)
        self.max_memory = max_memory
        
        # Enhanced action space with more specific actions
        self.action_space = [
            "scan_file",
            "quarantine_file", 
            "delete_file",
            "ignore_file",
            "update_database",
            "monitor_system",
            "heuristic_analysis",
            "deep_scan",
            "backup_file",
            "report_threat"
        ]
        
        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Experience replay memory
        self.memory = deque(maxlen=max_memory)
        
        # Q-table with default values
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Statistics and tracking
        self.actions_taken = []
        self.rewards = []
        self.state_history = []
        self.training_episodes = 0
        self.last_state = None
        self.last_action = None
        
        # Load existing model if available
        self._load_model()
        
        logger.info("Enhanced RL Agent initialized with model persistence")
    
    def _load_model(self):
        """Load the RL model from file with enhanced structure."""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'r') as f:
                    data = json.load(f)
                    
                    # Load Q-table
                    if "q_table" in data:
                        for state, actions in data["q_table"].items():
                            self.q_table[state] = defaultdict(float, actions)
                    
                    # Load statistics
                    self.actions_taken = data.get("actions_taken", [])[-1000:]  # Keep last 1000
                    self.rewards = data.get("rewards", [])[-1000:]
                    self.training_episodes = data.get("training_episodes", 0)
                    self.epsilon = data.get("epsilon", self.epsilon)
                    
                    # Load experience memory
                    if "memory" in data:
                        self.memory = deque(data["memory"][-1000:], maxlen=self.max_memory)
                    
                logger.info(f"RL model loaded: {self.training_episodes} episodes, epsilon={self.epsilon:.3f}")
            else:
                logger.info("No existing RL model found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading RL model: {e}")
            self.q_table = defaultdict(lambda: defaultdict(float))
    
    def _save_model(self):
        """Save the enhanced RL model to file."""
        try:
            # Convert defaultdict to regular dict for JSON serialization
            q_table_serializable = {}
            for state, actions in self.q_table.items():
                q_table_serializable[state] = dict(actions)
            
            data = {
                "q_table": q_table_serializable,
                "actions_taken": self.actions_taken[-1000:],
                "rewards": self.rewards[-1000:],
                "training_episodes": self.training_episodes,
                "epsilon": self.epsilon,
                "memory": list(self.memory)[-1000:],  # Convert deque to list
                "last_updated": time.time(),
                "model_version": "2.0"
            }
            
            # Create backup of existing model
            if self.model_path.exists():
                backup_path = self.model_path.with_suffix('.bak')
                self.model_path.rename(backup_path)
            
            with open(self.model_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"RL model saved: {self.training_episodes} episodes")
        except Exception as e:
            logger.error(f"Error saving RL model: {e}")
    
    def get_state(self, context: Dict[str, Any]) -> str:
        """
        Convert context to a comprehensive state representation.
        
        Args:
            context: Current context information
        
        Returns:
            State string representation
        """
        state_features = []
        
        # File-related features
        file_size = context.get("file_size", 0)
        if file_size == 0:
            state_features.append("empty_file")
        elif file_size < 1024:
            state_features.append("small_file")
        elif file_size < 1024 * 1024:
            state_features.append("medium_file")
        else:
            state_features.append("large_file")
        
        # Scan results
        scan_results = context.get("scan_results", {})
        if scan_results.get("infected", False):
            state_features.append("infected")
        elif scan_results.get("suspicious", False):
            risk_score = scan_results.get("risk_score", 0)
            if risk_score > 70:
                state_features.append("high_risk")
            elif risk_score > 30:
                state_features.append("medium_risk")
            else:
                state_features.append("low_risk")
        else:
            state_features.append("clean")
        
        # File type based on extension
        filepath = context.get("filepath", "")
        if filepath:
            extension = Path(filepath).suffix.lower()
            if extension in ['.exe', '.dll', '.bat', '.cmd']:
                state_features.append("executable")
            elif extension in ['.txt', '.log', '.md']:
                state_features.append("text_file")
            elif extension in ['.zip', '.tar', '.gz', '.rar']:
                state_features.append("archive")
            else:
                state_features.append("other_file")
        
        # System context
        system_load = context.get("system_load", 0)
        if system_load > 0.8:
            state_features.append("high_system_load")
        elif system_load > 0.5:
            state_features.append("medium_system_load")
        else:
            state_features.append("low_system_load")
        
        # Time-based features
        hour = time.localtime().tm_hour
        if 9 <= hour <= 17:
            state_features.append("business_hours")
        else:
            state_features.append("off_hours")
        
        return "_".join(sorted(state_features))
    
    def choose_action(self, state: str) -> str:
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state representation
            
        Returns:
            Selected action
        """
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            action = random.choice(self.action_space)
            logger.debug(f"Exploration: chose random action {action}")
        else:
            # Choose action with highest Q-value
            q_values = self.q_table[state]
            if not q_values:
                # If no Q-values exist, choose randomly
                action = random.choice(self.action_space)
            else:
                action = max(q_values.items(), key=lambda x: x[1])[0]
            logger.debug(f"Exploitation: chose best action {action}")
        
        return action
    
    def get_recommendations(self, context: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Get action recommendations based on current context.
        
        Args:
            context: Current context information
            top_k: Number of top recommendations to return
            
        Returns:
            List of recommended actions with confidence scores
        """
        state = self.get_state(context)
        q_values = self.q_table[state]
        
        # If no learned Q-values, use heuristic recommendations
        if not q_values:
            return self._get_heuristic_recommendations(context, top_k)
        
        # Sort actions by Q-value
        sorted_actions = sorted(q_values.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for action, q_value in sorted_actions[:top_k]:
            # Convert Q-value to confidence (0-1 range)
            max_q = max(q_values.values()) if q_values.values() else 1.0
            min_q = min(q_values.values()) if q_values.values() else 0.0
            
            if max_q == min_q:
                confidence = 0.5
            else:
                confidence = (q_value - min_q) / (max_q - min_q)
            
            recommendations.append({
                "action": self._format_action_description(action, context),
                "confidence": max(0.1, confidence),  # Minimum confidence of 0.1
                "q_value": q_value,
                "reasoning": self._get_action_reasoning(action, context)
            })
        
        return recommendations
    
    def _get_heuristic_recommendations(self, context: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """Generate heuristic recommendations when no learned data is available."""
        recommendations = []
        scan_results = context.get("scan_results", {})
        
        if scan_results.get("infected", False):
            recommendations.extend([
                {
                    "action": "Quarantine the infected file immediately",
                    "confidence": 0.9,
                    "reasoning": "File is confirmed infected by antivirus scanner"
                },
                {
                    "action": "Update virus database and rescan",
                    "confidence": 0.7,
                    "reasoning": "Ensure latest threat definitions are used"
                },
                {
                    "action": "Report threat to security team",
                    "confidence": 0.6,
                    "reasoning": "Document security incident for analysis"
                }
            ])
        elif scan_results.get("suspicious", False):
            risk_score = scan_results.get("risk_score", 0)
            if risk_score > 70:
                recommendations.extend([
                    {
                        "action": "Perform deep analysis of suspicious file",
                        "confidence": 0.8,
                        "reasoning": f"High risk score ({risk_score}%) requires thorough investigation"
                    },
                    {
                        "action": "Temporarily quarantine for further analysis",
                        "confidence": 0.6,
                        "reasoning": "Isolate potentially dangerous file"
                    }
                ])
            else:
                recommendations.extend([
                    {
                        "action": "Monitor file activity for unusual behavior",
                        "confidence": 0.7,
                        "reasoning": f"Medium risk score ({risk_score}%) warrants monitoring"
                    },
                    {
                        "action": "Perform additional heuristic checks",
                        "confidence": 0.5,
                        "reasoning": "Gather more information about suspicious indicators"
                    }
                ])
        else:
            recommendations.extend([
                {
                    "action": "File appears clean, no action needed",
                    "confidence": 0.8,
                    "reasoning": "No threats detected by multiple scan methods"
                },
                {
                    "action": "Continue regular monitoring",
                    "confidence": 0.6,
                    "reasoning": "Maintain baseline security monitoring"
                }
            ])
        
        return recommendations[:top_k]
    
    def _format_action_description(self, action: str, context: Dict[str, Any]) -> str:
        """Format action into human-readable description."""
        descriptions = {
            "scan_file": "Perform comprehensive file scan",
            "quarantine_file": "Move file to quarantine",
            "delete_file": "Permanently delete the file",
            "ignore_file": "Mark file as safe and ignore",
            "update_database": "Update threat database",
            "monitor_system": "Increase system monitoring",
            "heuristic_analysis": "Run detailed heuristic analysis",
            "deep_scan": "Perform deep behavioral analysis",
            "backup_file": "Create backup before action",
            "report_threat": "Report to security team"
        }
        return descriptions.get(action, action.replace("_", " ").title())
    
    def _get_action_reasoning(self, action: str, context: Dict[str, Any]) -> str:
        """Get reasoning for why an action is recommended."""
        scan_results = context.get("scan_results", {})
        
        if action == "quarantine_file":
            if scan_results.get("infected"):
                return "File is confirmed infected"
            elif scan_results.get("suspicious"):
                return "Suspicious behavior detected, isolation recommended"
            else:
                return "Precautionary measure"
        elif action == "scan_file":
            return "Comprehensive analysis needed"
        elif action == "delete_file":
            return "File poses significant threat"
        elif action == "ignore_file":
            return "File appears safe based on analysis"
        else:
            return f"Action {action} recommended by learning algorithm"
    
    def learn(self, state: str, action: str, reward: float, next_state: str):
        """
        Update Q-values using Q-learning algorithm.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
        """
        # Q-learning update rule
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        # Store experience in memory
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "timestamp": time.time()
        }
        self.memory.append(experience)
        
        # Update statistics
        self.actions_taken.append(action)
        self.rewards.append(reward)
        self.state_history.append(state)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Increment training episodes
        self.training_episodes += 1
        
        # Periodic model saving
        if self.training_episodes % 100 == 0:
            self._save_model()
        
        log_operation(logger, "RL_LEARN", {
            "state": state,
            "action": action,
            "reward": reward,
            "new_q_value": new_q,
            "epsilon": self.epsilon
        })
    
    def replay_experience(self, batch_size: int = 32):
        """
        Replay experiences from memory to improve learning.
        
        Args:
            batch_size: Number of experiences to replay
        """
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(list(self.memory), batch_size)
        
        for experience in batch:
            state = experience["state"]
            action = experience["action"]
            reward = experience["reward"]
            next_state = experience["next_state"]
            
            # Apply Q-learning update
            current_q = self.q_table[state][action]
            max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
            
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
            self.q_table[state][action] = new_q
        
        logger.debug(f"Replayed {batch_size} experiences")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            "training_episodes": self.training_episodes,
            "epsilon": self.epsilon,
            "memory_size": len(self.memory),
            "q_table_size": len(self.q_table),
                         "average_reward": (np.mean(self.rewards) if NUMPY_AVAILABLE else sum(self.rewards)/len(self.rewards)) if self.rewards else 0.0,
            "recent_actions": self.actions_taken[-10:] if self.actions_taken else [],
            "model_path": str(self.model_path)
        }
    
    def reset_model(self):
        """Reset the RL model to start fresh."""
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.memory.clear()
        self.actions_taken.clear()
        self.rewards.clear()
        self.state_history.clear()
        self.training_episodes = 0
        self.epsilon = 0.1
        
        # Remove model file
        if self.model_path.exists():
            self.model_path.unlink()
        
        logger.info("RL model reset to initial state")
