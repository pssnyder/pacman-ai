"""
Pacman Expert System - Rule-based AI player

This module implements a rule-based expert system that plays Pacman using specific movement rules:
1. If dot in front, continue eating dots in that direction
2. If red ghost ahead, turn around (flee opposite direction)
3. If blue ghost ahead, chase it (can eat vulnerable ghosts)
4. If apple (power pellet) visible, go towards it
5. If no dots on either side, start widening search pattern
6. When dots split in two directions, prefer current path unless red ghost ahead
7. Remember to turn at corners

The expert system serves as a baseline for comparing future AI performance.
"""

from typing import List, Tuple, Dict, Optional, Set
from collections import deque
import time
import math

from pacman_game import (
    PacmanGame, GameConfig, Map, Position, Direction, GameState, TileType,
    TurtleRenderer, ConsoleRenderer, NullRenderer,
    get_classic_map, get_simple_map
)


class PathFinder:
    """Pathfinding utilities for navigation"""
    
    @staticmethod
    def bfs_distance(start: Position, targets: Set[Position], game_map: Map, 
                     max_depth: int = 100) -> Optional[Tuple[Position, int, Direction]]:
        """
        Find nearest target using BFS
        
        Args:
            start: Starting position
            targets: Set of target positions
            game_map: Game map
            max_depth: Maximum search depth
            
        Returns:
            Tuple of (target_position, distance, first_direction) or None
        """
        if not targets:
            return None
        
        if start in targets:
            return (start, 0, Direction.NONE)
        
        queue = deque([(start, 0, Direction.NONE)])
        visited = {start}
        
        while queue:
            pos, dist, first_dir = queue.popleft()
            
            if dist >= max_depth:
                continue
            
            for neighbor, direction in game_map.get_neighbors(pos):
                if neighbor in visited:
                    continue
                
                visited.add(neighbor)
                
                # Track first direction taken from start
                next_first_dir = first_dir if first_dir != Direction.NONE else direction
                
                # Check if we found a target
                if neighbor in targets:
                    return (neighbor, dist + 1, next_first_dir)
                
                queue.append((neighbor, dist + 1, next_first_dir))
        
        return None
    
    @staticmethod
    def widening_search(start: Position, game_map: Map, max_radius: int = 10) -> List[Tuple[Position, int, Direction]]:
        """
        Perform widening search from start position, returning positions by distance
        
        Args:
            start: Starting position
            game_map: Game map
            max_radius: Maximum search radius
            
        Returns:
            List of tuples (position, distance, first_direction) sorted by distance
        """
        queue = deque([(start, 0, Direction.NONE)])
        visited = {start: (0, Direction.NONE)}
        results = []
        
        while queue:
            pos, dist, first_dir = queue.popleft()
            
            if dist >= max_radius:
                continue
            
            for neighbor, direction in game_map.get_neighbors(pos):
                if neighbor in visited:
                    continue
                
                next_first_dir = first_dir if first_dir != Direction.NONE else direction
                visited[neighbor] = (dist + 1, next_first_dir)
                results.append((neighbor, dist + 1, next_first_dir))
                queue.append((neighbor, dist + 1, next_first_dir))
        
        return sorted(results, key=lambda x: x[1])


class ExpertAgent:
    """Rule-based expert system for playing Pacman with specific movement rules"""
    
    def __init__(self, danger_threshold: int = 3, pellet_search_depth: int = 100):
        """
        Initialize expert agent
        
        Args:
            danger_threshold: Distance at which ghosts are considered dangerous
            pellet_search_depth: Maximum depth for pellet search
        """
        self.danger_threshold = danger_threshold
        self.pellet_search_depth = pellet_search_depth
        self.pathfinder = PathFinder()
        
        # Decision statistics
        self.stats = {
            'follow_dots_decisions': 0,
            'flee_red_ghost_decisions': 0,
            'chase_blue_ghost_decisions': 0,
            'seek_power_pellet_decisions': 0,
            'search_dots_decisions': 0,
            'turn_corner_decisions': 0,
            'total_decisions': 0
        }
    
    def reset_stats(self):
        """Reset decision statistics"""
        for key in self.stats:
            self.stats[key] = 0
    
    def check_position_for_pellet(self, pos: Position, game_map: Map) -> Tuple[bool, bool]:
        """
        Check if position has a pellet
        
        Returns:
            Tuple of (has_pellet, is_power_pellet)
        """
        tile = game_map.get_tile(pos)
        if tile == TileType.PELLET:
            return True, False
        elif tile == TileType.POWER_PELLET:
            return True, True
        return False, False
    
    def find_ghost_in_direction(self, pacman_pos: Position, direction: Direction, 
                                ghost_positions: List[Position], ghost_vulnerable: List[bool], 
                                game_map: Map, max_distance: int = 5) -> Optional[Tuple[Position, bool, int]]:
        """
        Check if there's a ghost in the given direction within max_distance
        
        Returns:
            Tuple of (ghost_pos, is_vulnerable, distance) or None
        """
        current_pos = pacman_pos
        for distance in range(1, max_distance + 1):
            current_pos = current_pos + direction
            if not game_map.is_walkable(current_pos):
                break
            
            for i, ghost_pos in enumerate(ghost_positions):
                if ghost_pos == current_pos:
                    return (ghost_pos, ghost_vulnerable[i], distance)
        
        return None
    
    def get_dots_in_direction(self, pacman_pos: Position, direction: Direction, 
                             game_map: Map, max_distance: int = 10) -> List[Tuple[Position, int]]:
        """
        Get all dots in a straight line in the given direction
        
        Returns:
            List of tuples (position, distance)
        """
        dots = []
        current_pos = pacman_pos
        
        for distance in range(1, max_distance + 1):
            current_pos = current_pos + direction
            if not game_map.is_walkable(current_pos):
                break
            
            has_pellet, is_power = self.check_position_for_pellet(current_pos, game_map)
            if has_pellet:
                dots.append((current_pos, distance))
        
        return dots
    
    def choose_action(self, game_state: Dict, current_direction: Direction, 
                     game_map: Map) -> Direction:
        """
        Choose next action based on specific expert rules
        
        Expert Rules (in priority order):
        1. Check all directions for red ghosts → avoid dangerous directions
        2. If blue ghost visible → chase it
        3. If power pellet visible → go towards it
        4. If dot in front in current direction → continue eating dots
        5. At corners/intersections, check for dots in other directions, prefer not turning
        6. If no dots nearby → widening search pattern
        
        Args:
            game_state: Current game state dictionary
            current_direction: Current direction of movement
            game_map: Game map
            
        Returns:
            Direction to move
        """
        self.stats['total_decisions'] += 1
        
        # Extract state information
        pacman_pos = Position(*game_state['pacman_pos'])
        ghost_positions = [Position(*gp) for gp in game_state['ghost_positions']]
        ghost_vulnerable = game_state['ghost_vulnerable']
        pellet_positions = set(Position(*pp) for pp in game_state['pellet_positions'])
        power_pellet_positions = set(Position(*pp) for pp in game_state['power_pellet_positions'])
        valid_action_names = game_state['valid_actions']
        valid_actions = [Direction[name] for name in valid_action_names]
        
        if not valid_actions:
            return Direction.NONE
        
        # Rule 1: Check ALL directions for red ghosts and filter out dangerous moves
        safe_actions = []
        dangerous_actions = []
        blue_ghost_directions = []
        
        for direction in valid_actions:
            ghost_in_dir = self.find_ghost_in_direction(
                pacman_pos, direction, ghost_positions, ghost_vulnerable, game_map, max_distance=self.danger_threshold
            )
            
            if ghost_in_dir:
                ghost_pos, is_vulnerable, distance = ghost_in_dir
                if is_vulnerable:
                    # Blue ghost - can chase it
                    blue_ghost_directions.append((direction, distance))
                    safe_actions.append(direction)
                else:
                    # Red ghost - dangerous!
                    dangerous_actions.append((direction, distance))
            else:
                # No ghost in this direction - safe
                safe_actions.append(direction)
        
        # If all directions are dangerous, we're trapped - pick the one with furthest ghost
        if not safe_actions:
            if dangerous_actions:
                # Pick direction with ghost furthest away
                dangerous_actions.sort(key=lambda x: -x[1])
                self.stats['flee_red_ghost_decisions'] += 1
                return dangerous_actions[0][0]
            # Should never happen, but fallback
            return valid_actions[0]
        
        # If we have blue ghosts to chase and we're in safe directions, go for them
        if blue_ghost_directions:
            # Pick closest blue ghost
            blue_ghost_directions.sort(key=lambda x: x[1])
            self.stats['chase_blue_ghost_decisions'] += 1
            return blue_ghost_directions[0][0]
        # If we have blue ghosts to chase and we're in safe directions, go for them
        if blue_ghost_directions:
            # Pick closest blue ghost
            blue_ghost_directions.sort(key=lambda x: x[1])
            self.stats['chase_blue_ghost_decisions'] += 1
            return blue_ghost_directions[0][0]
        
        # From here on, only consider safe directions
        # Rule 2: If power pellet visible in a safe direction, go towards it
        if power_pellet_positions:
            result = self.pathfinder.bfs_distance(pacman_pos, power_pellet_positions, game_map, max_depth=15)
            if result:
                _, distance, direction = result
                if distance <= 10 and direction in safe_actions:
                    self.stats['seek_power_pellet_decisions'] += 1
                    return direction
        
        # Rule 3: If dot in front in current direction and it's safe, continue eating
        if current_direction != Direction.NONE and current_direction in safe_actions:
            dots_ahead = self.get_dots_in_direction(pacman_pos, current_direction, game_map, max_distance=5)
            if dots_ahead:
                self.stats['follow_dots_decisions'] += 1
                return current_direction
        
        # Rule 4: At intersection - check safe directions for dots, prefer current direction
        dot_directions = []
        for direction in safe_actions:
            dots = self.get_dots_in_direction(pacman_pos, direction, game_map, max_distance=5)
            if dots:
                dot_directions.append((direction, len(dots), dots[0][1]))
        
        if dot_directions:
            # Prefer current direction if it has dots and is safe
            for direction, count, dist in dot_directions:
                if direction == current_direction:
                    self.stats['turn_corner_decisions'] += 1
                    return direction
            
            # Otherwise pick safe direction with most dots, breaking ties by closest dot
            dot_directions.sort(key=lambda x: (-x[1], x[2]))
            self.stats['turn_corner_decisions'] += 1
            return dot_directions[0][0]
        
        # Rule 5: No dots nearby - widening search pattern (only in safe directions)
        if pellet_positions:
            result = self.pathfinder.bfs_distance(
                pacman_pos, pellet_positions, game_map, max_depth=self.pellet_search_depth
            )
            
            if result:
                _, _, direction = result
                if direction in safe_actions:
                    self.stats['search_dots_decisions'] += 1
                    return direction
        
        # Fallback: Pick a safe direction, prefer current if possible
        if current_direction in safe_actions:
            return current_direction
        return safe_actions[0] if safe_actions else valid_actions[0]


class GameRunner:
    """Run games with expert system and track performance"""
    
    def __init__(self, visualize: bool = True, renderer_type: str = 'turtle'):
        """
        Initialize game runner
        
        Args:
            visualize: Whether to render the game
            renderer_type: Type of renderer ('turtle', 'console', or 'none')
        """
        self.visualize = visualize
        self.renderer_type = renderer_type
        
    def run_single_game(self, game: PacmanGame, expert: ExpertAgent, 
                       max_steps: int = 2000, tick_delay: int = 100) -> Dict:
        """
        Run a single game with expert system
        
        Args:
            game: PacmanGame instance
            expert: ExpertAgent instance
            max_steps: Maximum steps before timeout
            tick_delay: Delay between ticks in ms (0 for headless)
            
        Returns:
            Dictionary with game results
        """
        # Create renderer
        if self.visualize:
            if self.renderer_type == 'turtle':
                renderer = TurtleRenderer(tile_size=20)
            elif self.renderer_type == 'console':
                renderer = ConsoleRenderer(show_state_info=True)
            else:
                renderer = NullRenderer()
        else:
            renderer = NullRenderer()
        
        game.reset()
        game.start()
        expert.reset_stats()
        
        steps = 0
        start_time = time.time()
        total_reward = 0
        
        current_direction = Direction.NONE
        
        # Initial render
        if self.visualize:
            renderer.render(game)
        
        while steps < max_steps:
            # Get game state
            state = game.get_state()
            
            # Check if game is over
            if state['game_state'] in ['won', 'lost']:
                break
            
            # Expert chooses action
            action = expert.choose_action(state, current_direction, game.map)
            current_direction = action
            
            # Execute action
            _, reward, done, info = game.step(action)
            total_reward += reward
            
            # Render
            if self.visualize:
                renderer.render(game)
                if tick_delay > 0 and self.renderer_type == 'turtle':
                    time.sleep(tick_delay / 1000.0)
            
            steps += 1
            
            if done:
                break
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Clean up renderer
        if self.visualize and self.renderer_type != 'turtle':
            renderer.close()
        
        # Compile results
        final_state = game.get_state()
        result = {
            'won': final_state['game_state'] == 'won',
            'score': final_state['score'],
            'steps': steps,
            'time': elapsed_time,
            'pellets_remaining': final_state['pellet_count'],
            'lives_remaining': final_state['lives'],
            'timeout': steps >= max_steps,
            'total_reward': total_reward,
            'stats': expert.stats.copy()
        }
        
        return result
    
    def run_multiple_games(self, num_games: int, game_config: GameConfig, 
                          game_map: Map, expert: ExpertAgent, 
                          max_steps: int = 2000) -> Dict:
        """
        Run multiple games and aggregate statistics
        
        Args:
            num_games: Number of games to run
            game_config: Game configuration
            game_map: Game map
            expert: ExpertAgent instance
            max_steps: Maximum steps per game
            
        Returns:
            Dictionary with aggregated statistics
        """
        results = []
        
        print(f"\nRunning {num_games} games with expert system...")
        print("=" * 60)
        
        for i in range(num_games):
            # Create fresh game
            game = PacmanGame(game_config, game_map)
            
            # Run game (headless for batch testing)
            self.visualize = False
            result = self.run_single_game(game, expert, max_steps, tick_delay=0)
            results.append(result)
            
            # Print progress
            status = "WON" if result['won'] else "LOST"
            print(f"Game {i+1}/{num_games}: {status} | Score: {result['score']} | "
                  f"Steps: {result['steps']} | Time: {result['time']:.2f}s")
        
        # Aggregate statistics
        wins = sum(1 for r in results if r['won'])
        avg_score = sum(r['score'] for r in results) / len(results)
        avg_steps = sum(r['steps'] for r in results) / len(results)
        avg_time = sum(r['time'] for r in results) / len(results)
        avg_reward = sum(r['total_reward'] for r in results) / len(results)
        
        print("\n" + "=" * 60)
        print("AGGREGATE STATISTICS")
        print("=" * 60)
        print(f"Win Rate: {wins}/{num_games} ({100*wins/num_games:.1f}%)")
        print(f"Average Score: {avg_score:.1f}")
        print(f"Average Steps: {avg_steps:.1f}")
        print(f"Average Time: {avg_time:.2f}s")
        print(f"Average Reward: {avg_reward:.1f}")
        print(f"Games per Second: {num_games/sum(r['time'] for r in results):.2f}")
        
        # Decision statistics
        total_stats = {key: sum(r['stats'][key] for r in results) for key in results[0]['stats']}
        print("\nDECISION BREAKDOWN")
        print("=" * 60)
        for key, value in total_stats.items():
            if key != 'total_decisions' and total_stats['total_decisions'] > 0:
                pct = 100 * value / total_stats['total_decisions']
                print(f"{key}: {value} ({pct:.1f}%)")
        
        return {
            'results': results,
            'wins': wins,
            'win_rate': wins / num_games,
            'avg_score': avg_score,
            'avg_steps': avg_steps,
            'avg_time': avg_time,
            'avg_reward': avg_reward,
            'total_stats': total_stats
        }


def main():
    """Main entry point - run expert system demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pacman Expert System')
    parser.add_argument('--mode', choices=['single', 'batch'], default='single',
                       help='Run mode: single game or batch testing')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Visualize the game')
    parser.add_argument('--renderer', choices=['turtle', 'console', 'none'], default='turtle',
                       help='Renderer type')
    parser.add_argument('--games', type=int, default=10,
                       help='Number of games for batch mode')
    parser.add_argument('--map', choices=['classic', 'simple'], default='classic',
                       help='Map to use')
    parser.add_argument('--danger', type=int, default=3,
                       help='Danger threshold for ghost avoidance')
    parser.add_argument('--max-steps', type=int, default=2000,
                       help='Maximum steps per game')
    parser.add_argument('--tick-delay', type=int, default=50,
                       help='Delay between ticks in ms (for visualization)')
    
    args = parser.parse_args()
    
    # Create map
    if args.map == 'simple':
        game_map = get_simple_map()
        config = GameConfig(
            map_width=10,
            map_height=10,
            pacman_start=Position(5, 5),
            ghost_starts=[Position(1, 1), Position(8, 8)],
            ghost_count=2,
            ghost_ai_types=['random', 'chase']
        )
    else:
        game_map = get_classic_map()
        config = GameConfig(
            pacman_start=Position(8, 11),
            ghost_starts=[
                Position(7, 4),
                Position(7, 14),
                Position(12, 4),
                Position(12, 14)
            ],
            ghost_ai_types=['random', 'random', 'chase', 'random']  # Mixed difficulty
        )
    
    # Create expert
    expert = ExpertAgent(danger_threshold=args.danger)
    
    # Create runner
    runner = GameRunner(visualize=args.visualize, renderer_type=args.renderer)
    
    if args.mode == 'single':
        # Run single game
        print(f"\nRunning single game with expert system...")
        print(f"Map: {args.map} | Danger threshold: {args.danger}")
        print("=" * 60)
        
        game = PacmanGame(config, game_map)
        result = runner.run_single_game(game, expert, args.max_steps, args.tick_delay)
        
        print("\n" + "=" * 60)
        print("GAME RESULTS")
        print("=" * 60)
        print(f"Result: {'WON' if result['won'] else 'LOST'}")
        print(f"Score: {result['score']}")
        print(f"Steps: {result['steps']}")
        print(f"Time: {result['time']:.2f}s")
        print(f"Total Reward: {result['total_reward']}")
        print(f"Pellets Remaining: {result['pellets_remaining']}")
        print(f"Lives Remaining: {result['lives_remaining']}")
        
        print("\nDECISION BREAKDOWN")
        print("=" * 60)
        for key, value in result['stats'].items():
            if key != 'total_decisions' and result['stats']['total_decisions'] > 0:
                pct = 100 * value / result['stats']['total_decisions']
                print(f"{key}: {value} ({pct:.1f}%)")
    
    else:
        # Run batch testing
        stats = runner.run_multiple_games(
            args.games, config, game_map, expert, args.max_steps
        )


if __name__ == '__main__':
    main()
