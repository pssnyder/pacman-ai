"""
Pacman AI - Neural Network with Reinforcement Learning

This module implements a Deep Q-Learning neural network that learns to play Pacman
through experience. The AI has full "arcade player" perspective with vision-based
inputs covering the entire game map.

Features:
- Vision system: 360° proximity + extended cardinal vision + quadrant awareness
- Deep Q-Network with experience replay
- Comprehensive reward function
- Live play training with episode management
- Performance tracking vs expert baseline
- Model checkpointing and visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
from collections import deque, namedtuple
from typing import List, Tuple, Dict, Optional
import json
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from pacman_game import (
    PacmanGame, GameConfig, Map, Position, Direction, GameState, TileType,
    TurtleRenderer, ConsoleRenderer, NullRenderer,
    get_classic_map, get_simple_map
)

# Experience tuple for replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class VisionSystem:
    """Extract vision-based features from game state"""
    
    def __init__(self, map_width: int, map_height: int, 
                 proximity_range: int = 3, extended_range: int = 7):
        """
        Initialize vision system
        
        Args:
            map_width: Width of game map
            map_height: Height of game map
            proximity_range: Range for 360° proximity vision
            extended_range: Range for cardinal direction extended vision
        """
        self.map_width = map_width
        self.map_height = map_height
        self.proximity_range = proximity_range
        self.extended_range = extended_range
        
        # Calculate quadrant divisions (3x3 grid)
        self.quadrant_cols = 3
        self.quadrant_rows = 3
        self.quadrant_width = map_width / self.quadrant_cols
        self.quadrant_height = map_height / self.quadrant_rows
    
    def get_proximity_vision(self, pacman_pos: Position, game_state: Dict, 
                            game_map: Map) -> List[float]:
        """
        Get 8 vision inputs for 360° immediate proximity
        
        Returns 8 values (one per direction: N, NE, E, SE, S, SW, W, NW)
        Each value encodes: 0=wall, 0.25=empty, 0.5=pellet, 0.75=power_pellet, 
                           -0.5=red_ghost, -1.0=blue_ghost
        """
        directions = [
            Direction.UP,          # N
            (1, -1),              # NE
            Direction.RIGHT,       # E
            (1, 1),               # SE
            Direction.DOWN,        # S
            (-1, 1),              # SW
            Direction.LEFT,        # W
            (-1, -1),             # NW
        ]
        
        ghost_positions = [Position(*gp) for gp in game_state['ghost_positions']]
        ghost_vulnerable = game_state['ghost_vulnerable']
        
        vision = []
        for d in directions:
            if isinstance(d, Direction):
                dx, dy = d.value
            else:
                dx, dy = d
            
            # Look in this direction up to proximity_range
            found_value = 0.25  # default: empty
            for dist in range(1, self.proximity_range + 1):
                check_pos = Position(pacman_pos.x + dx * dist, pacman_pos.y + dy * dist)
                
                if not game_map.is_in_bounds(check_pos):
                    found_value = 0.0  # wall/boundary
                    break
                
                # Check for ghosts first
                for i, ghost_pos in enumerate(ghost_positions):
                    if ghost_pos == check_pos:
                        found_value = -1.0 if ghost_vulnerable[i] else -0.5
                        break
                
                if found_value < 0:  # Found ghost
                    break
                
                # Check tiles
                tile = game_map.get_tile(check_pos)
                if tile == TileType.WALL:
                    found_value = 0.0
                    break
                elif tile == TileType.POWER_PELLET:
                    found_value = 0.75
                    break
                elif tile == TileType.PELLET:
                    found_value = 0.5
                    # Don't break, keep looking for more important things
            
            vision.append(found_value)
        
        return vision
    
    def get_extended_vision(self, pacman_pos: Position, game_state: Dict, 
                           game_map: Map) -> List[float]:
        """
        Get 16 extended vision inputs for cardinal directions (N, E, S, W)
        Returns: [distance_to_wall, has_pellet, has_power_pellet, has_ghost] x 4 directions
        """
        directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        ghost_positions = [Position(*gp) for gp in game_state['ghost_positions']]
        
        vision = []
        for direction in directions:
            dx, dy = direction.value
            distance = 0
            has_pellet = 0.0
            has_power_pellet = 0.0
            has_ghost = 0.0
            
            for dist in range(1, self.extended_range + 1):
                check_pos = Position(pacman_pos.x + dx * dist, pacman_pos.y + dy * dist)
                
                if not game_map.is_in_bounds(check_pos) or game_map.get_tile(check_pos) == TileType.WALL:
                    distance = dist
                    break
                
                # Check for ghost
                if any(check_pos == gp for gp in ghost_positions):
                    has_ghost = 1.0
                
                # Check for pellets
                tile = game_map.get_tile(check_pos)
                if tile == TileType.PELLET:
                    has_pellet = 1.0
                elif tile == TileType.POWER_PELLET:
                    has_power_pellet = 1.0
                
                if dist == self.extended_range:
                    distance = dist
            
            # Normalize distance
            norm_distance = distance / self.extended_range if distance > 0 else 1.0
            vision.extend([norm_distance, has_pellet, has_power_pellet, has_ghost])
        
        return vision
    
    def get_quadrant_stats(self, pacman_pos: Position, game_state: Dict, 
                          game_map: Map) -> List[float]:
        """
        Get 9 quadrant statistics (3x3 grid)
        Each quadrant returns: [ghost_count, pellet_percentage, has_power_pellet]
        Total: 27 values (9 quadrants × 3 stats)
        """
        ghost_positions = [Position(*gp) for gp in game_state['ghost_positions']]
        pellet_positions = set(Position(*pp) for pp in game_state['pellet_positions'])
        power_pellet_positions = set(Position(*pp) for pp in game_state['power_pellet_positions'])
        
        stats = []
        
        for row in range(self.quadrant_rows):
            for col in range(self.quadrant_cols):
                # Define quadrant boundaries
                x_start = int(col * self.quadrant_width)
                x_end = int((col + 1) * self.quadrant_width)
                y_start = int(row * self.quadrant_height)
                y_end = int((row + 1) * self.quadrant_height)
                
                # Count ghosts in quadrant
                ghost_count = sum(1 for gp in ghost_positions 
                                 if x_start <= gp.x < x_end and y_start <= gp.y < y_end)
                
                # Count pellets in quadrant
                pellet_count = sum(1 for pp in pellet_positions 
                                  if x_start <= pp.x < x_end and y_start <= pp.y < y_end)
                
                # Count power pellets in quadrant
                power_count = sum(1 for pp in power_pellet_positions 
                                 if x_start <= pp.x < x_end and y_start <= pp.y < y_end)
                
                # Calculate quadrant size for normalization
                quadrant_size = (x_end - x_start) * (y_end - y_start)
                pellet_density = pellet_count / quadrant_size if quadrant_size > 0 else 0.0
                
                stats.extend([
                    ghost_count / 4.0,  # Normalize (max 4 ghosts)
                    min(pellet_density * 10, 1.0),  # Normalize pellet density
                    1.0 if power_count > 0 else 0.0  # Binary: has power pellet
                ])
        
        return stats
    
    def get_ghost_info(self, pacman_pos: Position, game_state: Dict) -> List[float]:
        """
        Get ghost-specific information
        Returns: [closest_red_ghost_dist, closest_blue_ghost_dist, 
                 red_approaching, blue_approaching, any_vulnerable]
        """
        ghost_positions = [Position(*gp) for gp in game_state['ghost_positions']]
        ghost_vulnerable = game_state['ghost_vulnerable']
        
        red_ghosts = [gp for i, gp in enumerate(ghost_positions) if not ghost_vulnerable[i]]
        blue_ghosts = [gp for i, gp in enumerate(ghost_positions) if ghost_vulnerable[i]]
        
        # Closest red ghost distance
        if red_ghosts:
            closest_red_dist = min(pacman_pos.manhattan_distance_to(gp) for gp in red_ghosts)
            closest_red_dist = min(closest_red_dist / 20.0, 1.0)  # Normalize
        else:
            closest_red_dist = 1.0
        
        # Closest blue ghost distance
        if blue_ghosts:
            closest_blue_dist = min(pacman_pos.manhattan_distance_to(gp) for gp in blue_ghosts)
            closest_blue_dist = min(closest_blue_dist / 20.0, 1.0)  # Normalize
        else:
            closest_blue_dist = 1.0
        
        # Are ghosts approaching? (simplified: just check if any are close)
        red_approaching = 1.0 if red_ghosts and closest_red_dist < 0.3 else 0.0
        blue_approaching = 1.0 if blue_ghosts and closest_blue_dist < 0.3 else 0.0
        any_vulnerable = 1.0 if any(ghost_vulnerable) else 0.0
        
        return [closest_red_dist, closest_blue_dist, red_approaching, blue_approaching, any_vulnerable]
    
    def get_self_awareness(self, pacman_pos: Position, game_state: Dict) -> List[float]:
        """
        Get Pacman's own state information
        Returns: [x_position, y_position, direction_one_hot(4), map_progress]
        Total: 7 values
        """
        # Normalize position
        norm_x = pacman_pos.x / self.map_width
        norm_y = pacman_pos.y / self.map_height
        
        # Direction one-hot encoding
        direction_name = game_state['pacman_direction']
        direction_map = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3, 'NONE': -1}
        dir_idx = direction_map.get(direction_name, -1)
        direction_one_hot = [1.0 if i == dir_idx else 0.0 for i in range(4)]
        
        # Map progress (pellets collected)
        total_pellets = len(game_state['pellet_positions']) + len(game_state['power_pellet_positions'])
        max_pellets = 196  # Approximate for classic map
        map_progress = 1.0 - (total_pellets / max_pellets) if max_pellets > 0 else 1.0
        
        return [norm_x, norm_y] + direction_one_hot + [map_progress]
    
    def extract_features(self, game_state: Dict, game_map: Map) -> np.ndarray:
        """
        Extract all features from game state
        
        Returns:
            NumPy array of features (total 63 values)
        """
        pacman_pos = Position(*game_state['pacman_pos'])
        
        features = []
        features.extend(self.get_proximity_vision(pacman_pos, game_state, game_map))  # 8
        features.extend(self.get_extended_vision(pacman_pos, game_state, game_map))   # 16
        features.extend(self.get_quadrant_stats(pacman_pos, game_state, game_map))    # 27
        features.extend(self.get_ghost_info(pacman_pos, game_state))                  # 5
        features.extend(self.get_self_awareness(pacman_pos, game_state))              # 7
        
        return np.array(features, dtype=np.float32)


class DQNetwork(nn.Module):
    """Deep Q-Network for Pacman decision making"""
    
    def __init__(self, input_size: int, output_size: int):
        """
        Initialize neural network
        
        Args:
            input_size: Number of input features
            output_size: Number of actions (4 for UP, DOWN, LEFT, RIGHT)
        """
        super(DQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass through network"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation on output (Q-values can be any real number)
        return x


class RewardFunction:
    """Calculate rewards for state transitions"""
    
    def __init__(self):
        # Reward weights
        self.pellet_reward = 1.0
        self.power_pellet_reward = 10.0
        self.ghost_eaten_reward = 200.0
        self.survival_reward = 0.1  # Small reward per step alive
        self.exploration_reward = 0.5  # Reward for visiting new areas
        
        # Penalties
        self.death_penalty = -500.0
        self.ghost_proximity_penalty = -0.5  # Per close ghost
        self.missed_blue_ghost_penalty = -50.0  # Had vulnerable ghost but didn't eat
        self.repetitive_movement_penalty = -0.2  # Moving in empty areas
        self.time_penalty = -0.05  # Small penalty per step to encourage efficiency
        
        # Tracking
        self.visited_positions = set()
        self.last_pellet_count = 0
        self.last_power_pellets = set()
        self.vulnerable_timer = 0
        self.blue_ghosts_eaten_during_vulnerable = 0
    
    def reset(self):
        """Reset tracking for new episode"""
        self.visited_positions = set()
        self.last_pellet_count = 0
        self.last_power_pellets = set()
        self.vulnerable_timer = 0
        self.blue_ghosts_eaten_during_vulnerable = 0
    
    def calculate_reward(self, old_state: Dict, action: Direction, new_state: Dict, 
                        done: bool, info: Dict) -> float:
        """
        Calculate reward for state transition
        
        Args:
            old_state: Previous game state
            action: Action taken
            new_state: Resulting game state
            done: Whether episode ended
            info: Additional info from game
            
        Returns:
            Total reward value
        """
        reward = 0.0
        
        pacman_pos = tuple(new_state['pacman_pos'])
        old_pellet_count = len(old_state['pellet_positions']) + len(old_state['power_pellet_positions'])
        new_pellet_count = len(new_state['pellet_positions']) + len(new_state['power_pellet_positions'])
        old_power_pellets = set(tuple(pp) for pp in old_state['power_pellet_positions'])
        new_power_pellets = set(tuple(pp) for pp in new_state['power_pellet_positions'])
        
        # Pellet collection
        if new_pellet_count < old_pellet_count:
            pellets_collected = old_pellet_count - new_pellet_count
            
            # Check if power pellet was collected
            if len(new_power_pellets) < len(old_power_pellets):
                reward += self.power_pellet_reward
                self.vulnerable_timer = 50  # Start tracking vulnerable period
                self.blue_ghosts_eaten_during_vulnerable = 0
            else:
                reward += self.pellet_reward * pellets_collected
        
        # Ghost eaten (detect by score increase beyond pellets)
        old_score = old_state['score']
        new_score = new_state['score']
        score_diff = new_score - old_score
        
        if score_diff >= 200:  # Ghost eaten gives 200 points
            reward += self.ghost_eaten_reward
            self.blue_ghosts_eaten_during_vulnerable += 1
        
        # Survival reward
        if not done:
            reward += self.survival_reward
        
        # Exploration reward
        if pacman_pos not in self.visited_positions:
            self.visited_positions.add(pacman_pos)
            reward += self.exploration_reward
        
        # Ghost proximity penalty
        ghost_positions = [tuple(gp) for gp in new_state['ghost_positions']]
        ghost_vulnerable = new_state['ghost_vulnerable']
        for i, ghost_pos in enumerate(ghost_positions):
            dist = abs(pacman_pos[0] - ghost_pos[0]) + abs(pacman_pos[1] - ghost_pos[1])
            if dist <= 3 and not ghost_vulnerable[i]:  # Close red ghost
                reward += self.ghost_proximity_penalty
        
        # Update vulnerable timer
        if any(new_state['ghost_vulnerable']):
            if self.vulnerable_timer > 0:
                self.vulnerable_timer -= 1
        else:
            # Vulnerability ended - check if we ate blue ghosts
            if self.vulnerable_timer > 0 and self.blue_ghosts_eaten_during_vulnerable == 0:
                reward += self.missed_blue_ghost_penalty  # Missed opportunity!
            self.vulnerable_timer = 0
        
        # Repetitive movement penalty (moving in area with no pellets)
        tile_has_pellet = any(pacman_pos == tuple(pp) for pp in old_state['pellet_positions'])
        tile_has_power = any(pacman_pos == tuple(pp) for pp in old_state['power_pellet_positions'])
        if not tile_has_pellet and not tile_has_power:
            reward += self.repetitive_movement_penalty
        
        # Death penalty
        if done:
            reason = info.get('reason', '')
            if reason == 'lost_all_lives' or reason == 'life_lost':
                reward += self.death_penalty
            elif reason == 'won':
                reward += 500.0  # Big bonus for winning!
        
        # Time penalty
        reward += self.time_penalty
        
        return reward


class PacmanDQNAgent:
    """DQN Agent for learning to play Pacman"""
    
    def __init__(self, state_size: int, action_size: int, config: Dict):
        """
        Initialize DQN agent
        
        Args:
            state_size: Size of state vector
            action_size: Number of possible actions
            config: Configuration dictionary
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get('use_cuda', False) else "cpu")
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)  # Discount factor
        self.epsilon = config.get('epsilon_start', 1.0)  # Exploration rate
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 64)
        self.memory_size = config.get('memory_size', 10000)
        
        # Networks
        self.policy_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Experience replay memory
        self.memory = deque(maxlen=self.memory_size)
        
        # Training stats
        self.steps_done = 0
        self.episodes_done = 0
    
    def select_action(self, state: np.ndarray, valid_actions: List[Direction]) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state features
            valid_actions: List of valid actions
            
        Returns:
            Action index
        """
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Random valid action
            action_idx = random.randint(0, self.action_size - 1)
        else:
            # Greedy action from policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.argmax().item()
        
        self.steps_done += 1
        return action_idx
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def train_step(self):
        """Perform one training step (experience replay)"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Save model and training state"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
        }, filepath)
    
    def load(self, filepath: str):
        """Load model and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.steps_done = checkpoint.get('steps_done', 0)
        self.episodes_done = checkpoint.get('episodes_done', 0)


# Direction mapping
ACTION_TO_DIRECTION = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
DIRECTION_TO_ACTION = {d: i for i, d in enumerate(ACTION_TO_DIRECTION)}


class PacmanTrainer:
    """Training loop for Pacman AI"""
    
    def __init__(self, config: Dict):
        """Initialize trainer with configuration"""
        self.config = config
        
        # Create game map
        if config.get('map_type', 'classic') == 'simple':
            self.game_map = get_simple_map()
            self.game_config = GameConfig(
                map_width=10,
                map_height=10,
                pacman_start=Position(5, 5),
                ghost_starts=[Position(1, 1), Position(8, 8)],
                ghost_count=2,
                ghost_ai_types=['random', 'chase']
            )
        else:
            self.game_map = get_classic_map()
            self.game_config = GameConfig(
                pacman_start=Position(8, 11),
                ghost_starts=[
                    Position(7, 4),
                    Position(7, 14),
                    Position(12, 4),
                    Position(12, 14)
                ],
                ghost_ai_types=config.get('ghost_ai_types', ['random', 'random', 'chase', 'random'])
            )
        
        # Initialize vision system
        self.vision = VisionSystem(self.game_config.map_width, self.game_config.map_height)
        
        # Initialize agent
        state_size = 63  # Total features from vision system
        action_size = 4  # UP, DOWN, LEFT, RIGHT
        self.agent = PacmanDQNAgent(state_size, action_size, config)
        
        # Reward function
        self.reward_func = RewardFunction()
        
        # Training statistics
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_steps = []
        self.episode_wins = []
        self.losses = []
    
    def train(self, num_episodes: int, max_steps: int = 2000, 
             save_interval: int = 100, visualize_interval: int = 0):
        """
        Train the agent
        
        Args:
            num_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            save_interval: Save checkpoint every N episodes
            visualize_interval: Visualize game every N episodes (0 = never)
        """
        print(f"\nStarting training for {num_episodes} episodes...")
        print(f"Device: {self.agent.device}")
        print(f"State size: {self.agent.state_size}, Action size: {self.agent.action_size}")
        print("=" * 70)
        
        for episode in range(num_episodes):
            # Create game
            game = PacmanGame(self.game_config, self.game_map)
            game.reset()
            game.start()
            self.reward_func.reset()
            
            # Get initial state
            game_state = game.get_state()
            state = self.vision.extract_features(game_state, self.game_map)
            
            episode_reward = 0
            episode_loss = []
            step = 0
            
            # Visualize occasionally
            renderer = None
            if visualize_interval > 0 and episode % visualize_interval == 0:
                renderer = TurtleRenderer(tile_size=20)
            
            while step < max_steps:
                # Select action
                action_idx = self.agent.select_action(state, game.get_valid_actions(game.pacman.position))
                action = ACTION_TO_DIRECTION[action_idx]
                
                # Execute action
                old_game_state = game_state
                _, _, done, info = game.step(action)
                game_state = game.get_state()
                next_state = self.vision.extract_features(game_state, self.game_map)
                
                # Calculate reward
                reward = self.reward_func.calculate_reward(old_game_state, action, game_state, done, info)
                episode_reward += reward
                
                # Store experience
                self.agent.store_experience(state, action_idx, reward, next_state, done)
                
                # Train
                loss = self.agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)
                
                # Visualize
                if renderer:
                    renderer.render(game)
                    time.sleep(0.05)
                
                state = next_state
                step += 1
                
                if done:
                    break
            
            # Update target network periodically
            if episode % 10 == 0:
                self.agent.update_target_network()
            
            # Decay epsilon
            self.agent.decay_epsilon()
            self.agent.episodes_done += 1
            
            # Record statistics
            final_state = game.get_state()
            self.episode_rewards.append(episode_reward)
            self.episode_scores.append(final_state['score'])
            self.episode_steps.append(step)
            self.episode_wins.append(1 if final_state['game_state'] == 'won' else 0)
            if episode_loss:
                self.losses.append(np.mean(episode_loss))
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_score = np.mean(self.episode_scores[-10:])
                win_rate = np.mean(self.episode_wins[-10:]) * 100
                print(f"Episode {episode}/{num_episodes} | "
                      f"Reward: {episode_reward:.1f} (avg: {avg_reward:.1f}) | "
                      f"Score: {final_state['score']} (avg: {avg_score:.1f}) | "
                      f"Steps: {step} | Win Rate: {win_rate:.0f}% | "
                      f"Epsilon: {self.agent.epsilon:.3f}")
            
            # Save checkpoint
            if save_interval > 0 and episode > 0 and episode % save_interval == 0:
                self.save_checkpoint(f"checkpoint_ep{episode}.pth")
                self.save_stats(f"stats_ep{episode}.json")
            
            # Clean up renderer
            if renderer:
                renderer.close()
        
        print("\n" + "=" * 70)
        print("Training complete!")
        self.save_checkpoint("final_model.pth")
        self.save_stats("final_stats.json")
        self.plot_training_progress()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        self.agent.save(filename)
        print(f"Saved checkpoint: {filename}")
    
    def save_stats(self, filename: str):
        """Save training statistics"""
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_scores': self.episode_scores,
            'episode_steps': self.episode_steps,
            'episode_wins': self.episode_wins,
            'losses': self.losses
        }
        with open(filename, 'w') as f:
            json.dump(stats, f)
    
    def plot_training_progress(self):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        
        # Scores
        axes[0, 1].plot(self.episode_scores)
        axes[0, 1].set_title('Episode Scores')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Score')
        
        # Win rate (rolling average)
        window = 50
        if len(self.episode_wins) >= window:
            win_rate = np.convolve(self.episode_wins, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(win_rate)
        axes[1, 0].set_title(f'Win Rate (Rolling {window})')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Win Rate')
        
        # Loss
        if self.losses:
            axes[1, 1].plot(self.losses)
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        print("Saved training plot: training_progress.png")
        plt.close()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pacman AI Training')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--map', choices=['classic', 'simple'], default='classic', help='Map to use')
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'], default='medium',
                       help='Ghost difficulty')
    parser.add_argument('--visualize', type=int, default=0, help='Visualize every N episodes (0=never)')
    parser.add_argument('--save-interval', type=int, default=100, help='Save checkpoint every N episodes')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--load', type=str, help='Load model from checkpoint')
    
    args = parser.parse_args()
    
    # Ghost AI configuration
    if args.difficulty == 'easy':
        ghost_ai_types = ['random', 'random', 'random', 'random']
    elif args.difficulty == 'medium':
        ghost_ai_types = ['random', 'random', 'chase', 'random']
    else:  # hard
        ghost_ai_types = ['chase', 'chase', 'chase', 'chase']
    
    # Training configuration
    config = {
        'map_type': args.map,
        'ghost_ai_types': ghost_ai_types,
        'use_cuda': args.cuda,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'learning_rate': 0.001,
        'batch_size': 64,
        'memory_size': 10000,
    }
    
    # Create trainer
    trainer = PacmanTrainer(config)
    
    # Load checkpoint if specified
    if args.load:
        trainer.agent.load(args.load)
        print(f"Loaded model from {args.load}")
    
    # Train
    trainer.train(
        num_episodes=args.episodes,
        max_steps=2000,
        save_interval=args.save_interval,
        visualize_interval=args.visualize
    )


if __name__ == '__main__':
    main()
