"""
CCTV-Style Game Viewer - Watch Expert vs AI play Pacman side-by-side

Multi-panel visualization with auto-restart and speed control.
"""

# CRITICAL: Set backend BEFORE any matplotlib imports
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List, Dict, Tuple
import argparse
import torch
import torch.nn as nn

from pacman_game import (
    PacmanGame, GameConfig, Map, Position, Direction, TileType,
    NullRenderer, get_classic_map
)
from pacman_expert import ExpertAgent

# Import AI components without importing matplotlib again
import sys
import importlib.util

# Load DQNetwork class directly to avoid matplotlib re-import
class DQNetwork(nn.Module):
    """Deep Q-Network for Pacman decision making"""
    
    def __init__(self, input_size: int, output_size: int):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        import torch.nn.functional as F
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class VisionSystem:
    """Extract vision-based features from game state"""
    
    def __init__(self, map_width: int, map_height: int, 
                 proximity_range: int = 3, extended_range: int = 7):
        self.map_width = map_width
        self.map_height = map_height
        self.proximity_range = proximity_range
        self.extended_range = extended_range
        self.quadrant_cols = 3
        self.quadrant_rows = 3
        self.quadrant_width = map_width / self.quadrant_cols
        self.quadrant_height = map_height / self.quadrant_rows
    
    def get_proximity_vision(self, pacman_pos: Position, game_state: Dict, 
                            game_map: Map) -> List[float]:
        directions = [
            Direction.UP, (1, -1), Direction.RIGHT, (1, 1),
            Direction.DOWN, (-1, 1), Direction.LEFT, (-1, -1),
        ]
        
        ghost_positions = [Position(*gp) for gp in game_state['ghost_positions']]
        ghost_vulnerable = game_state['ghost_vulnerable']
        
        vision = []
        for d in directions:
            if isinstance(d, Direction):
                dx, dy = d.value
            else:
                dx, dy = d
            
            found_value = 0.25
            for dist in range(1, self.proximity_range + 1):
                check_pos = Position(pacman_pos.x + dx * dist, pacman_pos.y + dy * dist)
                
                if not game_map.is_in_bounds(check_pos):
                    found_value = 0.0
                    break
                
                for i, ghost_pos in enumerate(ghost_positions):
                    if ghost_pos == check_pos:
                        found_value = -1.0 if ghost_vulnerable[i] else -0.5
                        break
                
                if found_value < 0:
                    break
                
                tile = game_map.get_tile(check_pos)
                if tile == TileType.WALL:
                    found_value = 0.0
                    break
                elif tile == TileType.POWER_PELLET:
                    found_value = 0.75
                    break
                elif tile == TileType.PELLET:
                    found_value = 0.5
            
            vision.append(found_value)
        
        return vision
    
    def get_extended_vision(self, pacman_pos: Position, game_state: Dict, 
                           game_map: Map) -> List[float]:
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
                
                if any(check_pos == gp for gp in ghost_positions):
                    has_ghost = 1.0
                
                tile = game_map.get_tile(check_pos)
                if tile == TileType.PELLET:
                    has_pellet = 1.0
                elif tile == TileType.POWER_PELLET:
                    has_power_pellet = 1.0
                
                if dist == self.extended_range:
                    distance = dist
            
            norm_distance = distance / self.extended_range if distance > 0 else 1.0
            vision.extend([norm_distance, has_pellet, has_power_pellet, has_ghost])
        
        return vision
    
    def get_quadrant_stats(self, pacman_pos: Position, game_state: Dict, 
                          game_map: Map) -> List[float]:
        ghost_positions = [Position(*gp) for gp in game_state['ghost_positions']]
        pellet_positions = set(Position(*pp) for pp in game_state['pellet_positions'])
        power_pellet_positions = set(Position(*pp) for pp in game_state['power_pellet_positions'])
        
        stats = []
        
        for row in range(self.quadrant_rows):
            for col in range(self.quadrant_cols):
                x_start = int(col * self.quadrant_width)
                x_end = int((col + 1) * self.quadrant_width)
                y_start = int(row * self.quadrant_height)
                y_end = int((row + 1) * self.quadrant_height)
                
                ghost_count = sum(1 for gp in ghost_positions 
                                 if x_start <= gp.x < x_end and y_start <= gp.y < y_end)
                
                pellet_count = sum(1 for pp in pellet_positions 
                                  if x_start <= pp.x < x_end and y_start <= pp.y < y_end)
                
                power_count = sum(1 for pp in power_pellet_positions 
                                 if x_start <= pp.x < x_end and y_start <= pp.y < y_end)
                
                quadrant_size = (x_end - x_start) * (y_end - y_start)
                pellet_density = pellet_count / quadrant_size if quadrant_size > 0 else 0.0
                
                stats.extend([
                    ghost_count / 4.0,
                    min(pellet_density * 10, 1.0),
                    1.0 if power_count > 0 else 0.0
                ])
        
        return stats
    
    def get_ghost_info(self, pacman_pos: Position, game_state: Dict) -> List[float]:
        ghost_positions = [Position(*gp) for gp in game_state['ghost_positions']]
        ghost_vulnerable = game_state['ghost_vulnerable']
        
        red_ghosts = [gp for i, gp in enumerate(ghost_positions) if not ghost_vulnerable[i]]
        blue_ghosts = [gp for i, gp in enumerate(ghost_positions) if ghost_vulnerable[i]]
        
        if red_ghosts:
            closest_red_dist = min(pacman_pos.manhattan_distance_to(gp) for gp in red_ghosts)
            closest_red_dist = min(closest_red_dist / 20.0, 1.0)
        else:
            closest_red_dist = 1.0
        
        if blue_ghosts:
            closest_blue_dist = min(pacman_pos.manhattan_distance_to(gp) for gp in blue_ghosts)
            closest_blue_dist = min(closest_blue_dist / 20.0, 1.0)
        else:
            closest_blue_dist = 1.0
        
        red_approaching = 1.0 if red_ghosts and closest_red_dist < 0.3 else 0.0
        blue_approaching = 1.0 if blue_ghosts and closest_blue_dist < 0.3 else 0.0
        any_vulnerable = 1.0 if any(ghost_vulnerable) else 0.0
        
        return [closest_red_dist, closest_blue_dist, red_approaching, blue_approaching, any_vulnerable]
    
    def get_self_awareness(self, pacman_pos: Position, game_state: Dict) -> List[float]:
        norm_x = pacman_pos.x / self.map_width
        norm_y = pacman_pos.y / self.map_height
        
        direction_name = game_state['pacman_direction']
        direction_map = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3, 'NONE': -1}
        dir_idx = direction_map.get(direction_name, -1)
        direction_one_hot = [1.0 if i == dir_idx else 0.0 for i in range(4)]
        
        total_pellets = len(game_state['pellet_positions']) + len(game_state['power_pellet_positions'])
        max_pellets = 196
        map_progress = 1.0 - (total_pellets / max_pellets) if max_pellets > 0 else 1.0
        
        return [norm_x, norm_y] + direction_one_hot + [map_progress]
    
    def extract_features(self, game_state: Dict, game_map: Map) -> np.ndarray:
        pacman_pos = Position(*game_state['pacman_pos'])
        
        features = []
        features.extend(self.get_proximity_vision(pacman_pos, game_state, game_map))
        features.extend(self.get_extended_vision(pacman_pos, game_state, game_map))
        features.extend(self.get_quadrant_stats(pacman_pos, game_state, game_map))
        features.extend(self.get_ghost_info(pacman_pos, game_state))
        features.extend(self.get_self_awareness(pacman_pos, game_state))
        
        return np.array(features, dtype=np.float32)


class SimpleAIAgent:
    """Simplified AI agent for visualization"""
    
    def __init__(self):
        self.device = torch.device("cpu")
        self.policy_net = DQNetwork(63, 4).to(self.device)
        self.epsilon = 0.0
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.policy_net.eval()
    
    def select_action(self, state: np.ndarray, valid_actions) -> int:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action_idx = q_values.argmax().item()
        return action_idx


ACTION_TO_DIRECTION = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


class GamePanel:
    """Single game panel in the viewer"""
    
    def __init__(self, ax, title: str, agent_type: str, difficulty: str, 
                 game_map: Map, game_config: GameConfig):
        """
        Initialize game panel
        
        Args:
            ax: Matplotlib axis for this panel
            title: Panel title (e.g., "Expert - Medium")
            agent_type: 'expert' or 'ai'
            difficulty: 'easy', 'medium', or 'hard'
            game_map: Game map to use
            game_config: Game configuration
        """
        self.ax = ax
        self.title = title
        self.agent_type = agent_type
        self.difficulty = difficulty
        self.game_map = game_map
        self.game_config = game_config
        
        # Initialize game
        self.game = PacmanGame(game_config, game_map)
        self.game.reset()
        self.game.start()
        
        # Initialize agent
        if agent_type == 'expert':
            self.agent = ExpertAgent()
            self.current_direction = Direction.NONE
        else:  # AI
            self.vision = VisionSystem(game_config.map_width, game_config.map_height)
            self.agent = SimpleAIAgent()
            try:
                self.agent.load('final_model.pth')
            except Exception as e:
                print(f"Warning: Could not load AI model for {title}: {e}")
        
        # Game state
        self.step_count = 0
        self.total_games = 0
        self.total_wins = 0
        self.total_score = 0
        
        # Visual setup
        self.setup_panel()
    
    def setup_panel(self):
        """Setup panel visualization"""
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.game_config.map_width - 0.5)
        self.ax.set_ylim(-0.5, self.game_config.map_height - 0.5)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()
        self.ax.set_title(self.title, fontsize=10, fontweight='bold')
        self.ax.axis('off')
    
    def render_game(self):
        """Render current game state"""
        self.ax.clear()
        self.setup_panel()
        
        state = self.game.get_state()
        
        # Draw walls
        for y in range(self.game_config.map_height):
            for x in range(self.game_config.map_width):
                pos = Position(x, y)
                tile = self.game_map.get_tile(pos)
                
                if tile == TileType.WALL:
                    rect = patches.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                                            linewidth=0, facecolor='#2121DE')
                    self.ax.add_patch(rect)
        
        # Draw pellets
        for px, py in state['pellet_positions']:
            circle = patches.Circle((px, py), 0.08, facecolor='#FFB897', edgecolor='none')
            self.ax.add_patch(circle)
        
        # Draw power pellets
        for px, py in state['power_pellet_positions']:
            circle = patches.Circle((px, py), 0.2, facecolor='orange', edgecolor='none')
            self.ax.add_patch(circle)
        
        # Draw ghosts
        ghost_positions = state['ghost_positions']
        ghost_vulnerable = state['ghost_vulnerable']
        ghost_colors = ['#FF0000', '#FFB8FF', '#00FFFF', '#FFB852']  # Red, Pink, Cyan, Orange
        
        for i, (gx, gy) in enumerate(ghost_positions):
            if i < len(ghost_vulnerable) and ghost_vulnerable[i]:
                color = '#4D4DFF'  # Blue when vulnerable
            else:
                color = ghost_colors[i % len(ghost_colors)]
            
            circle = patches.Circle((gx, gy), 0.35, facecolor=color, edgecolor='black', linewidth=0.5)
            self.ax.add_patch(circle)
        
        # Draw Pacman
        pacman_x, pacman_y = state['pacman_pos']
        pacman_circle = patches.Circle((pacman_x, pacman_y), 0.35, 
                                      facecolor='yellow', edgecolor='black', linewidth=1)
        self.ax.add_patch(pacman_circle)
        
        # Draw stats
        game_state = state['game_state']
        status_color = 'green' if game_state == 'won' else 'red' if game_state == 'lost' else 'white'
        stats_text = f"Score: {state['score']} | Steps: {self.step_count} | Games: {self.total_games}"
        if self.total_games > 0:
            win_rate = (self.total_wins / self.total_games) * 100
            avg_score = self.total_score / self.total_games
            stats_text += f"\nWR: {win_rate:.0f}% | Avg: {avg_score:.0f}"
        
        self.ax.text(self.game_config.map_width / 2, -0.2, stats_text,
                    ha='center', va='top', fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.3))
    
    def step(self):
        """Execute one game step"""
        state = self.game.get_state()
        
        # Check if game is over
        if state['game_state'] in ['won', 'lost']:
            # Record stats
            self.total_games += 1
            self.total_score += state['score']
            if state['game_state'] == 'won':
                self.total_wins += 1
            
            # Restart game
            self.game.reset()
            self.game.start()
            self.step_count = 0
            if self.agent_type == 'expert':
                self.current_direction = Direction.NONE
            return
        
        # Get action from agent
        if self.agent_type == 'expert':
            action = self.agent.choose_action(state, self.current_direction, self.game_map)
            self.current_direction = action
        else:  # AI
            features = self.vision.extract_features(state, self.game_map)
            action_idx = self.agent.select_action(features, self.game.get_valid_actions(self.game.pacman.position))
            action = ACTION_TO_DIRECTION[action_idx]
        
        # Execute action
        self.game.step(action)
        self.step_count += 1


class MultiPanelViewer:
    """Multi-panel game viewer"""
    
    def __init__(self, layout: str = '2-panel', speed: int = 2):
        """
        Initialize viewer
        
        Args:
            layout: '2-panel' or '6-panel'
            speed: Playback speed multiplier (1 = normal, 2 = 2x, etc.)
        """
        self.layout = layout
        self.speed = speed
        self.game_map = get_classic_map()
        
        # Setup figure
        if layout == '2-panel':
            self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 8))
            self.panels = self.create_2_panel_layout()
        else:  # 6-panel
            self.fig, self.axes = plt.subplots(3, 2, figsize=(16, 20))
            self.panels = self.create_6_panel_layout()
        
        self.fig.suptitle('Pacman AI Observatory - Live Game Monitoring', 
                         fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    def create_2_panel_layout(self) -> List[GamePanel]:
        """Create 2-panel layout: Expert (medium) vs AI (medium)"""
        panels = []
        
        # Medium difficulty config
        config = GameConfig(
            pacman_start=Position(8, 11),
            ghost_starts=[Position(7, 4), Position(7, 14), Position(12, 4), Position(12, 14)],
            ghost_ai_types=['random', 'random', 'chase', 'random']
        )
        
        # Expert panel
        panels.append(GamePanel(
            self.axes[0], 
            "Expert System - Medium Difficulty",
            'expert', 
            'medium',
            self.game_map,
            config
        ))
        
        # AI panel
        panels.append(GamePanel(
            self.axes[1],
            "Neural Network AI - Medium Difficulty",
            'ai',
            'medium',
            self.game_map,
            config
        ))
        
        return panels
    
    def create_6_panel_layout(self) -> List[GamePanel]:
        """Create 6-panel layout: 3 difficulties x 2 agents"""
        panels = []
        difficulties = [
            ('Easy', ['random', 'random', 'random', 'random']),
            ('Medium', ['random', 'random', 'chase', 'random']),
            ('Hard', ['chase', 'chase', 'chase', 'chase'])
        ]
        
        for row, (diff_name, ghost_types) in enumerate(difficulties):
            config = GameConfig(
                pacman_start=Position(8, 11),
                ghost_starts=[Position(7, 4), Position(7, 14), Position(12, 4), Position(12, 14)],
                ghost_ai_types=ghost_types
            )
            
            # Expert panel (left column)
            panels.append(GamePanel(
                self.axes[row, 0],
                f"Expert - {diff_name}",
                'expert',
                diff_name.lower(),
                self.game_map,
                config
            ))
            
            # AI panel (right column)
            panels.append(GamePanel(
                self.axes[row, 1],
                f"AI - {diff_name}",
                'ai',
                diff_name.lower(),
                self.game_map,
                config
            ))
        
        return panels
    
    def update(self, frame):
        """Update all panels (called by animation)"""
        # Step each panel 'speed' times per frame
        for _ in range(self.speed):
            for panel in self.panels:
                panel.step()
        
        # Render all panels
        for panel in self.panels:
            panel.render_game()
        
        return []
    
    def run(self):
        """Start the viewer"""
        print(f"\nStarting {self.layout} viewer at {self.speed}x speed...")
        print("Close the window to exit.")
        
        # Animation: update every 50ms (20 FPS)
        self.anim = FuncAnimation(
            self.fig, 
            self.update,
            interval=50,  # 50ms = 20 FPS
            blit=False,
            cache_frame_data=False
        )
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Watch Expert vs AI play Pacman')
    parser.add_argument('--layout', choices=['2-panel', '6-panel'], default='2-panel',
                       help='Layout: 2-panel (medium only) or 6-panel (all difficulties)')
    parser.add_argument('--speed', type=int, default=2,
                       help='Playback speed multiplier (1=normal, 2=2x, 3=3x, etc.)')
    
    args = parser.parse_args()
    
    viewer = MultiPanelViewer(layout=args.layout, speed=args.speed)
    viewer.run()


if __name__ == '__main__':
    main()
