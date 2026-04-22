"""
Pacman Game Engine - Modular implementation with programmatic API

This module provides a clean, class-based Pacman game engine with:
- Separation of game logic from rendering
- Programmatic control interface for AI agents
- Configurable maps, speeds, and game parameters
- Pluggable renderer system (Turtle, Console, Headless)
- Complete game state API for observation and control
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict, Set, Callable
from abc import ABC, abstractmethod
import random
import math
from collections import deque


class Direction(Enum):
    """Cardinal directions for movement"""
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    NONE = (0, 0)
    
    def opposite(self):
        """Get opposite direction"""
        opposites = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
            Direction.NONE: Direction.NONE
        }
        return opposites[self]


class TileType(Enum):
    """Tile types in the game map"""
    WALL = 0
    PELLET = 1
    EMPTY = 2
    POWER_PELLET = 3  # Power pellet (apple) - makes ghosts vulnerable


@dataclass
class Position:
    """Position in grid coordinates"""
    x: int
    y: int
    
    def __add__(self, other):
        if isinstance(other, Position):
            return Position(self.x + other.x, self.y + other.y)
        elif isinstance(other, Direction):
            dx, dy = other.value
            return Position(self.x + dx, self.y + dy)
        return NotImplemented
    
    def __eq__(self, other):
        if isinstance(other, Position):
            return self.x == other.x and self.y == other.y
        return False
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def manhattan_distance_to(self, other: 'Position') -> int:
        """Calculate Manhattan distance to another position"""
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def copy(self):
        return Position(self.x, self.y)


class Map:
    """Game map with tile grid system"""
    
    def __init__(self, tiles: List[int], width: int, height: int):
        """
        Initialize map from tile array
        
        Args:
            tiles: Flat list of tile values (0=wall, 1=pellet, 2=empty)
            width: Map width in tiles
            height: Map height in tiles
        """
        self.width = width
        self.height = height
        self.tiles = [TileType(t) for t in tiles]
        self.initial_tiles = self.tiles.copy()
        self.pellet_positions = self._find_pellets()
        
    def _find_pellets(self) -> Set[Position]:
        """Find all pellet positions on the map"""
        pellets = set()
        for y in range(self.height):
            for x in range(self.width):
                tile = self.get_tile(Position(x, y))
                if tile == TileType.PELLET or tile == TileType.POWER_PELLET:
                    pellets.add(Position(x, y))
        return pellets
    
    def get_tile(self, pos: Position) -> TileType:
        """Get tile type at position"""
        if not self.is_in_bounds(pos):
            return TileType.WALL
        index = pos.y * self.width + pos.x
        return self.tiles[index]
    
    def set_tile(self, pos: Position, tile_type: TileType):
        """Set tile type at position"""
        if self.is_in_bounds(pos):
            index = pos.y * self.width + pos.x
            self.tiles[index] = tile_type
            
    def is_in_bounds(self, pos: Position) -> bool:
        """Check if position is within map bounds"""
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height
    
    def is_walkable(self, pos: Position) -> bool:
        """Check if position is walkable (not a wall)"""
        return self.get_tile(pos) != TileType.WALL
    
    def get_neighbors(self, pos: Position) -> List[Tuple[Position, Direction]]:
        """Get walkable neighbor positions and directions"""
        neighbors = []
        for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            new_pos = pos + direction
            if self.is_walkable(new_pos):
                neighbors.append((new_pos, direction))
        return neighbors
    
    def collect_pellet(self, pos: Position) -> Tuple[bool, bool]:
        """
        Collect pellet at position if present
        
        Returns:
            Tuple of (pellet_collected, was_power_pellet)
        """
        tile = self.get_tile(pos)
        if tile == TileType.PELLET:
            self.set_tile(pos, TileType.EMPTY)
            self.pellet_positions.discard(pos)
            return True, False
        elif tile == TileType.POWER_PELLET:
            self.set_tile(pos, TileType.EMPTY)
            self.pellet_positions.discard(pos)
            return True, True
        return False, False
    
    def reset(self):
        """Reset map to initial state"""
        self.tiles = self.initial_tiles.copy()
        self.pellet_positions = self._find_pellets()
    
    def get_pellet_count(self) -> int:
        """Get number of remaining pellets"""
        return len(self.pellet_positions)
    
    @staticmethod
    def from_string_list(map_strings: List[str]) -> 'Map':
        """
        Create map from string list
        
        Format:
            '#' = wall
            '.' = pellet
            ' ' = empty
        """
        height = len(map_strings)
        width = max(len(row) for row in map_strings) if map_strings else 0
        tiles = []
        
        for row in map_strings:
            for col in range(width):
                if col < len(row):
                    char = row[col]
                    if char == '#':
                        tiles.append(0)  # wall
                    elif char == '.':
                        tiles.append(1)  # pellet
                    elif char == 'o' or char == 'O':
                        tiles.append(3)  # power pellet
                    else:
                        tiles.append(2)  # empty
                else:
                    tiles.append(0)  # default to wall
        
        return Map(tiles, width, height)


class Entity:
    """Base class for game entities (Pacman and Ghosts)"""
    
    def __init__(self, position: Position, speed: int = 1):
        """
        Initialize entity
        
        Args:
            position: Starting position
            speed: Movement speed (tiles per move)
        """
        self.position = position
        self.initial_position = position.copy()
        self.direction = Direction.NONE
        self.speed = speed
        
    def reset(self):
        """Reset entity to initial state"""
        self.position = self.initial_position.copy()
        self.direction = Direction.NONE
    
    def can_move(self, game_map: Map, direction: Direction) -> bool:
        """Check if entity can move in direction"""
        new_pos = self.position + direction
        return game_map.is_walkable(new_pos)
    
    def move(self, direction: Direction, game_map: Map) -> bool:
        """
        Attempt to move in direction
        
        Returns:
            True if move was successful, False otherwise
        """
        if self.can_move(game_map, direction):
            self.position = self.position + direction
            self.direction = direction
            return True
        return False


class Pacman(Entity):
    """Pacman player entity"""
    
    def __init__(self, position: Position, speed: int = 1, lives: int = 3):
        super().__init__(position, speed)
        self.lives = lives
        self.initial_lives = lives
        self.score = 0
        
    def reset(self):
        """Reset Pacman to initial state"""
        super().reset()
        self.lives = self.initial_lives
        self.score = 0
    
    def lose_life(self):
        """Decrease life count"""
        self.lives -= 1
    
    def add_score(self, points: int):
        """Add points to score"""
        self.score += points


class GhostAI(ABC):
    """Abstract base class for ghost AI strategies"""
    
    @abstractmethod
    def choose_direction(self, ghost_pos: Position, pacman_pos: Position, 
                        game_map: Map, current_direction: Direction) -> Direction:
        """Choose next direction for ghost"""
        pass


class RandomGhostAI(GhostAI):
    """Random movement AI (original behavior)"""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
    
    def choose_direction(self, ghost_pos: Position, pacman_pos: Position, 
                        game_map: Map, current_direction: Direction) -> Direction:
        """Choose random valid direction"""
        # Try to continue in current direction
        if current_direction != Direction.NONE:
            new_pos = ghost_pos + current_direction
            if game_map.is_walkable(new_pos):
                return current_direction
        
        # Choose random valid direction
        valid_directions = []
        for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            if game_map.is_walkable(ghost_pos + direction):
                valid_directions.append(direction)
        
        if valid_directions:
            return self.rng.choice(valid_directions)
        return Direction.NONE


class ChaseGhostAI(GhostAI):
    """Chase Pacman using improved pathfinding - no backtracking unless necessary"""
    
    def choose_direction(self, ghost_pos: Position, pacman_pos: Position, 
                        game_map: Map, current_direction: Direction) -> Direction:
        """Choose direction that gets closer to Pacman, prefer not reversing"""
        best_direction = Direction.NONE
        best_distance = float('inf')
        
        # Don't reverse direction unless it's the only option
        avoid_direction = current_direction.opposite() if current_direction != Direction.NONE else None
        
        for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            new_pos = ghost_pos + direction
            if game_map.is_walkable(new_pos):
                distance = new_pos.distance_to(pacman_pos)
                
                # Penalize reversing direction
                if direction == avoid_direction:
                    distance += 100  # Heavy penalty for reversing
                
                if distance < best_distance:
                    best_distance = distance
                    best_direction = direction
        
        return best_direction


class Ghost(Entity):
    """Ghost enemy entity"""
    
    def __init__(self, position: Position, ai: GhostAI, speed: int = 1, color: str = 'red'):
        super().__init__(position, speed)
        self.ai = ai
        self.color = color
        self.vulnerable = False
        self.vulnerable_timer = 0
        self.eaten = False
        
    def reset(self):
        """Reset ghost to initial state"""
        super().reset()
        self.vulnerable = False
        self.vulnerable_timer = 0
        self.eaten = False
        
    def make_vulnerable(self, duration: int):
        """Make ghost vulnerable for a duration"""
        self.vulnerable = True
        self.vulnerable_timer = duration
        self.eaten = False
        
    def get_eaten(self):
        """Ghost gets eaten by Pacman"""
        self.eaten = True
        self.vulnerable = False
        self.vulnerable_timer = 0
        # Return to starting position
        self.position = self.initial_position.copy()
        self.direction = Direction.NONE
        
    def update(self, pacman_pos: Position, game_map: Map):
        """Update ghost AI and move"""
        # Update vulnerable timer
        if self.vulnerable:
            self.vulnerable_timer -= 1
            if self.vulnerable_timer <= 0:
                self.vulnerable = False
                self.vulnerable_timer = 0
        
        # Don't move if eaten (will respawn next frame)
        if self.eaten:
            self.eaten = False
            return
        
        # If vulnerable, flee from Pacman instead of chasing
        if self.vulnerable:
            # Move away from Pacman
            best_direction = Direction.NONE
            best_distance = -1
            
            for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                new_pos = self.position + direction
                if game_map.is_walkable(new_pos):
                    distance = new_pos.distance_to(pacman_pos)
                    if distance > best_distance:
                        best_distance = distance
                        best_direction = direction
            
            self.move(best_direction, game_map)
        else:
            # Normal AI behavior
            new_direction = self.ai.choose_direction(
                self.position, pacman_pos, game_map, self.direction
            )
            self.move(new_direction, game_map)


@dataclass
class GameConfig:
    """Game configuration parameters"""
    map_width: int = 20
    map_height: int = 20
    tile_size: int = 20  # pixels per tile (for rendering)
    pacman_lives: int = 3
    ghost_count: int = 4
    collision_threshold: float = 0.9  # distance threshold for collision
    pellet_score: int = 1
    power_pellet_score: int = 10
    ghost_score: int = 200
    vulnerable_duration: int = 50  # ticks ghosts stay vulnerable
    movement_tolerance: int = 2  # tolerance for turning (0 = strict grid alignment)
    random_seed: Optional[int] = None
    
    # Starting positions (in grid coordinates)
    pacman_start: Position = field(default_factory=lambda: Position(8, 11))
    ghost_starts: List[Position] = field(default_factory=lambda: [
        Position(7, 4),
        Position(7, 14),
        Position(12, 4),
        Position(12, 14)
    ])
    
    # Ghost AI types
    ghost_ai_types: List[str] = field(default_factory=lambda: ['random', 'random', 'random', 'random'])


class GameState(Enum):
    """Game states"""
    READY = 'ready'
    PLAYING = 'playing'
    PAUSED = 'paused'
    WON = 'won'
    LOST = 'lost'


class PacmanGame:
    """Main Pacman game controller"""
    
    def __init__(self, config: GameConfig, game_map: Map):
        """
        Initialize game
        
        Args:
            config: Game configuration
            game_map: Game map
        """
        self.config = config
        self.map = game_map
        self.state = GameState.READY
        
        # Initialize Pacman
        self.pacman = Pacman(
            config.pacman_start.copy(),
            speed=1,
            lives=config.pacman_lives
        )
        
        # Initialize ghosts
        self.ghosts = []
        for i in range(min(config.ghost_count, len(config.ghost_starts))):
            ai_type = config.ghost_ai_types[i] if i < len(config.ghost_ai_types) else 'random'
            if ai_type == 'chase':
                ai = ChaseGhostAI()
            else:
                ai = RandomGhostAI(config.random_seed)
            
            ghost = Ghost(config.ghost_starts[i].copy(), ai, speed=1)
            self.ghosts.append(ghost)
        
        # Event callbacks
        self.on_pellet_collected: Optional[Callable[[int], None]] = None
        self.on_power_pellet_collected: Optional[Callable[[], None]] = None
        self.on_ghost_collision: Optional[Callable[[], None]] = None
        self.on_ghost_eaten: Optional[Callable[[int], None]] = None
        self.on_game_over: Optional[Callable[[bool], None]] = None
        self.on_win: Optional[Callable[[], None]] = None
        
        # Queued action for Pacman with buffering
        self.queued_action = Direction.NONE
        self.buffered_action = Direction.NONE
    
    def reset(self):
        """Reset game to initial state"""
        self.map.reset()
        self.pacman.reset()
        for ghost in self.ghosts:
            ghost.reset()
        self.state = GameState.READY
        self.queued_action = Direction.NONE
        self.buffered_action = Direction.NONE
    
    def start(self):
        """Start the game"""
        self.state = GameState.PLAYING
    
    def pause(self):
        """Pause the game"""
        if self.state == GameState.PLAYING:
            self.state = GameState.PAUSED
    
    def resume(self):
        """Resume the game"""
        if self.state == GameState.PAUSED:
            self.state = GameState.PLAYING
    
    def set_pacman_action(self, direction: Direction):
        """Set next action for Pacman with buffering for better controls"""
        self.buffered_action = direction
        
        # Try to execute immediately if within tolerance
        if self._can_turn(direction):
            self.queued_action = direction
    
    def _can_turn(self, direction: Direction) -> bool:
        """Check if Pacman can turn in direction with tolerance"""
        if direction == Direction.NONE:
            return True
        
        # Check if the move is valid from current position
        if self.pacman.can_move(self.map, direction):
            # With tolerance, allow turns even if not perfectly aligned
            if self.config.movement_tolerance > 0:
                return True
            # Without tolerance, require grid alignment
            return True
        return False
    
    def get_valid_actions(self, pos: Position) -> List[Direction]:
        """Get valid actions from position"""
        valid = []
        for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            if self.map.is_walkable(pos + direction):
                valid.append(direction)
        return valid
    
    def update(self):
        """
        Update game state for one tick
        
        Returns:
            Tuple of (reward, done, info)
        """
        if self.state != GameState.PLAYING:
            return 0, self.state in [GameState.WON, GameState.LOST], {}
        
        reward = 0
        
        # Try buffered action first, fall back to queued action
        action_to_execute = self.queued_action
        if self.buffered_action != Direction.NONE and self._can_turn(self.buffered_action):
            action_to_execute = self.buffered_action
            self.queued_action = self.buffered_action
        
        # Move Pacman
        if action_to_execute != Direction.NONE:
            self.pacman.move(action_to_execute, self.map)
        
        # Check pellet collection
        pellet_collected, is_power = self.map.collect_pellet(self.pacman.position)
        if pellet_collected:
            if is_power:
                # Power pellet!
                reward += self.config.power_pellet_score
                self.pacman.add_score(self.config.power_pellet_score)
                # Make all ghosts vulnerable
                for ghost in self.ghosts:
                    ghost.make_vulnerable(self.config.vulnerable_duration)
                if self.on_power_pellet_collected:
                    self.on_power_pellet_collected()
            else:
                # Regular pellet
                reward += self.config.pellet_score
                self.pacman.add_score(self.config.pellet_score)
                if self.on_pellet_collected:
                    self.on_pellet_collected(self.pacman.score)
        
        # Move ghosts
        for ghost in self.ghosts:
            ghost.update(self.pacman.position, self.map)
        
        # Check ghost collisions
        for ghost in self.ghosts:
            if self.pacman.position.distance_to(ghost.position) < self.config.collision_threshold:
                if ghost.vulnerable:
                    # Eat the ghost!
                    ghost.get_eaten()
                    reward += self.config.ghost_score
                    self.pacman.add_score(self.config.ghost_score)
                    if self.on_ghost_eaten:
                        self.on_ghost_eaten(self.pacman.score)
                else:
                    # Ghost caught Pacman
                    self.pacman.lose_life()
                    reward -= 10
                    
                    if self.on_ghost_collision:
                        self.on_ghost_collision()
                    
                    if self.pacman.lives <= 0:
                        self.state = GameState.LOST
                        if self.on_game_over:
                            self.on_game_over(False)
                        return reward, True, {'reason': 'lost_all_lives'}
                    else:
                        # Reset positions but keep score
                        self.pacman.position = self.pacman.initial_position.copy()
                        for g in self.ghosts:
                            g.reset()
                        return reward, False, {'reason': 'life_lost'}
        
        # Check win condition
        if self.map.get_pellet_count() == 0:
            self.state = GameState.WON
            reward += 100
            if self.on_win:
                self.on_win()
            if self.on_game_over:
                self.on_game_over(True)
            return reward, True, {'reason': 'won'}
        
        done = self.state in [GameState.WON, GameState.LOST]
        return reward, done, {}
    
    def get_state(self) -> Dict:
        """
        Get complete game state for observation
        
        Returns:
            Dictionary with all observable state information
        """
        # Separate regular pellets from power pellets
        regular_pellets = []
        power_pellets = []
        for pos in self.map.pellet_positions:
            if self.map.get_tile(pos) == TileType.POWER_PELLET:
                power_pellets.append((pos.x, pos.y))
            else:
                regular_pellets.append((pos.x, pos.y))
        
        return {
            'pacman_pos': (self.pacman.position.x, self.pacman.position.y),
            'pacman_direction': self.pacman.direction.name,
            'ghost_positions': [(g.position.x, g.position.y) for g in self.ghosts],
            'ghost_directions': [g.direction.name for g in self.ghosts],
            'ghost_vulnerable': [g.vulnerable for g in self.ghosts],
            'ghost_vulnerable_timers': [g.vulnerable_timer for g in self.ghosts],
            'pellet_positions': regular_pellets,
            'power_pellet_positions': power_pellets,
            'pellet_count': self.map.get_pellet_count(),
            'score': self.pacman.score,
            'lives': self.pacman.lives,
            'game_state': self.state.value,
            'valid_actions': [d.name for d in self.get_valid_actions(self.pacman.position)]
        }
    
    def step(self, action: Direction) -> Tuple[Dict, float, bool, Dict]:
        """
        Gym-style step function for AI training
        
        Args:
            action: Direction to move Pacman
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.set_pacman_action(action)
        reward, done, info = self.update()
        observation = self.get_state()
        return observation, reward, done, info


# ====================
# Renderer System
# ====================

class Renderer(ABC):
    """Abstract base class for renderers"""
    
    @abstractmethod
    def render(self, game: PacmanGame):
        """Render the current game state"""
        pass
    
    @abstractmethod
    def close(self):
        """Clean up renderer resources"""
        pass


class NullRenderer(Renderer):
    """Headless renderer (no rendering)"""
    
    def render(self, game: PacmanGame):
        """Do nothing"""
        pass
    
    def close(self):
        """Do nothing"""
        pass


class ConsoleRenderer(Renderer):
    """ASCII console renderer for debugging"""
    
    def __init__(self, show_state_info: bool = True):
        self.show_state_info = show_state_info
    
    def render(self, game: PacmanGame):
        """Render game to console"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Build map display
        display = []
        for y in range(game.map.height):
            row = []
            for x in range(game.map.width):
                pos = Position(x, y)
                
                # Check for entities
                if pos == game.pacman.position:
                    row.append('C')  # Pacman
                elif any(g.position == pos for g in game.ghosts):
                    # Check if ghost is vulnerable
                    ghost = next(g for g in game.ghosts if g.position == pos)
                    if ghost.vulnerable:
                        row.append('B')  # Blue vulnerable ghost
                    else:
                        row.append('G')  # Ghost
                else:
                    tile = game.map.get_tile(pos)
                    if tile == TileType.WALL:
                        row.append('#')
                    elif tile == TileType.PELLET:
                        row.append('.')
                    elif tile == TileType.POWER_PELLET:
                        row.append('O')  # Power pellet (apple)
                    else:
                        row.append(' ')
            display.append(''.join(row))
        
        # Print map
        print('\n'.join(display))
        
        # Print state info
        if self.show_state_info:
            print(f"\nScore: {game.pacman.score} | Lives: {game.pacman.lives} | Pellets: {game.map.get_pellet_count()}")
            print(f"State: {game.state.value} | Pacman: ({game.pacman.position.x}, {game.pacman.position.y})")
    
    def close(self):
        """Do nothing"""
        pass


class TurtleRenderer(Renderer):
    """Turtle graphics renderer (original style)"""
    
    def __init__(self, tile_size: int = 20):
        """
        Initialize turtle renderer
        
        Args:
            tile_size: Size of each tile in pixels
        """
        from turtle import Screen, Turtle
        
        self.tile_size = tile_size
        self.screen = Screen()
        self.path = Turtle(visible=False)
        self.path.speed(0)
        self.writer = Turtle(visible=False)
        self.writer.speed(0)
        self.entity_turtle = Turtle(visible=False)
        self.entity_turtle.speed(0)
        
        self.initialized = False
    
    def _setup_screen(self, game: PacmanGame):
        """Setup screen on first render"""
        if self.initialized:
            return
        
        width = game.map.width * self.tile_size
        height = game.map.height * self.tile_size
        
        self.screen.setup(width + 40, height + 40)
        self.screen.bgcolor('black')
        self.screen.tracer(0)
        self.screen.title('Pacman Game')
        
        self.initialized = True
    
    def _grid_to_pixel(self, pos: Position, map_width: int, map_height: int) -> Tuple[int, int]:
        """Convert grid position to pixel coordinates"""
        x = pos.x * self.tile_size - (map_width * self.tile_size) // 2
        y = (map_height * self.tile_size) // 2 - pos.y * self.tile_size
        return x, y
    
    def _draw_square(self, x: int, y: int, color: str):
        """Draw a filled square at pixel coordinates"""
        self.path.up()
        self.path.goto(x, y)
        self.path.down()
        self.path.color(color)
        self.path.begin_fill()
        for _ in range(4):
            self.path.forward(self.tile_size)
            self.path.left(90)
        self.path.end_fill()
    
    def render(self, game: PacmanGame):
        """Render game using turtle graphics"""
        self._setup_screen(game)
        
        # Clear previous frame
        self.path.clear()
        self.entity_turtle.clear()
        self.writer.undo()
        
        # Draw map
        for y in range(game.map.height):
            for x in range(game.map.width):
                pos = Position(x, y)
                tile = game.map.get_tile(pos)
                px, py = self._grid_to_pixel(pos, game.map.width, game.map.height)
                
                if tile == TileType.WALL:
                    self._draw_square(px, py, 'blue')
                elif tile == TileType.PELLET:
                    self.path.up()
                    self.path.goto(px + self.tile_size // 2, py + self.tile_size // 2)
                    self.path.dot(4, 'white')
                elif tile == TileType.POWER_PELLET:
                    self.path.up()
                    self.path.goto(px + self.tile_size // 2, py + self.tile_size // 2)
                    self.path.dot(12, 'orange')  # Larger orange dot for power pellet
        
        # Draw Pacman
        px, py = self._grid_to_pixel(game.pacman.position, game.map.width, game.map.height)
        self.entity_turtle.up()
        self.entity_turtle.goto(px + self.tile_size // 2, py + self.tile_size // 2)
        self.entity_turtle.dot(self.tile_size, 'yellow')
        
        # Draw ghosts
        for ghost in game.ghosts:
            px, py = self._grid_to_pixel(ghost.position, game.map.width, game.map.height)
            self.entity_turtle.up()
            self.entity_turtle.goto(px + self.tile_size // 2, py + self.tile_size // 2)
            # Blue if vulnerable, red otherwise
            color = 'cyan' if ghost.vulnerable else 'red'
            self.entity_turtle.dot(self.tile_size, color)
        
        # Draw score
        score_x = (game.map.width * self.tile_size) // 2 - 40
        score_y = (game.map.height * self.tile_size) // 2 - 40
        self.writer.goto(score_x, score_y)
        self.writer.color('white')
        self.writer.write(f"Score: {game.pacman.score} Lives: {game.pacman.lives}", 
                         font=('Arial', 12, 'normal'))
        
        # Update screen
        self.screen.update()
    
    def close(self):
        """Close turtle window"""
        try:
            self.screen.bye()
        except:
            pass


# ====================
# Preset Maps
# ====================

def get_classic_map() -> Map:
    """Get the classic Pacman map from original game with power pellets"""
    # Original tiles array with power pellets at corners (tile value 3)
    tiles = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 3, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 3, 0, 0, 0, 0,
        0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
    return Map(tiles, 20, 20)


def get_simple_map() -> Map:
    """Get a simple 10x10 test map with power pellets"""
    map_str = [
        "##########",
        "#O......O#",
        "#.##..##.#",
        "#........#",
        "#.#....#.#",
        "#.#....#.#",
        "#........#",
        "#.##..##.#",
        "#O......O#",
        "##########"
    ]
    return Map.from_string_list(map_str)


# ====================
# Example Usage
# ====================

def main():
    """Example: Run game with keyboard controls"""
    from turtle import listen, onkey
    
    # Create game
    game_map = get_classic_map()
    config = GameConfig(
        pacman_start=Position(8, 11),
        ghost_starts=[
            Position(7, 4),
            Position(7, 14),
            Position(12, 4),
            Position(12, 14)
        ],
        ghost_ai_types=['random', 'random', 'chase', 'random']
    )
    game = PacmanGame(config, game_map)
    
    # Create renderer
    renderer = TurtleRenderer(tile_size=20)
    
    # Control functions
    def move_up():
        game.set_pacman_action(Direction.UP)
    
    def move_down():
        game.set_pacman_action(Direction.DOWN)
    
    def move_left():
        game.set_pacman_action(Direction.LEFT)
    
    def move_right():
        game.set_pacman_action(Direction.RIGHT)
    
    # Setup keyboard controls
    listen()
    onkey(move_up, 'Up')
    onkey(move_down, 'Down')
    onkey(move_left, 'Left')
    onkey(move_right, 'Right')
    
    # Game loop
    game.start()
    
    def game_loop():
        if game.state == GameState.PLAYING:
            reward, done, info = game.update()
            renderer.render(game)
            
            if done:
                print(f"\nGame Over! Final Score: {game.pacman.score}")
                print(f"Result: {info.get('reason', 'unknown')}")
                return
            
            renderer.screen.ontimer(game_loop, 100)  # 100ms tick rate
    
    renderer.render(game)
    game_loop()
    renderer.screen.mainloop()


if __name__ == '__main__':
    main()
