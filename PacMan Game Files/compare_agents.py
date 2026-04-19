"""
Compare Expert System vs Neural Network AI Performance
"""

import torch
import numpy as np
from typing import Dict, List
import json

from pacman_game import (
    PacmanGame, GameConfig, Map, Position, Direction, TileType,
    NullRenderer, get_classic_map
)
from pacman_expert import ExpertAgent
from pacman_ai import PacmanDQNAgent, VisionSystem, ACTION_TO_DIRECTION


class AgentComparator:
    """Compare performance of different agents"""
    
    def __init__(self, config: Dict):
        self.config = config
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
    
    def evaluate_expert(self, num_games: int = 10) -> Dict:
        """Evaluate expert system"""
        print(f"\nEvaluating Expert System ({num_games} games)...")
        print("-" * 60)
        
        expert = ExpertAgent()
        scores = []
        rewards = []
        steps_list = []
        wins = 0
        
        for i in range(num_games):
            game = PacmanGame(self.game_config, self.game_map)
            game.reset()
            game.start()
            
            total_reward = 0
            step = 0
            max_steps = 2000
            current_direction = Direction.NONE
            
            while step < max_steps:
                state = game.get_state()
                action = expert.choose_action(state, current_direction, self.game_map)
                current_direction = action
                
                _, reward, done, info = game.step(action)
                total_reward += reward
                step += 1
                
                if done:
                    break
            
            final_state = game.get_state()
            scores.append(final_state['score'])
            rewards.append(total_reward)
            steps_list.append(step)
            
            if final_state['game_state'] == 'won':
                wins += 1
            
            if (i + 1) % 5 == 0:
                print(f"Game {i+1}/{num_games}: Score={final_state['score']}, Steps={step}, "
                      f"Status={'WIN' if final_state['game_state'] == 'won' else 'LOST'}")
        
        results = {
            'agent': 'Expert',
            'scores': scores,
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'avg_reward': np.mean(rewards),
            'avg_steps': np.mean(steps_list),
            'wins': wins,
            'win_rate': wins / num_games * 100
        }
        
        self._print_results(results)
        return results
    
    def evaluate_ai(self, model_path: str, num_games: int = 10) -> Dict:
        """Evaluate trained neural network"""
        print(f"\nEvaluating Neural Network AI ({num_games} games)...")
        print("-" * 60)
        
        # Load AI
        vision = VisionSystem(self.game_config.map_width, self.game_config.map_height)
        agent = PacmanDQNAgent(63, 4, {'epsilon_start': 0.0})  # No exploration during eval
        agent.load(model_path)
        agent.epsilon = 0.0  # Pure exploitation
        
        scores = []
        rewards = []
        steps_list = []
        wins = 0
        
        for i in range(num_games):
            game = PacmanGame(self.game_config, self.game_map)
            game.reset()
            game.start()
            
            state = vision.extract_features(game.get_state(), self.game_map)
            total_reward = 0
            step = 0
            max_steps = 2000
            
            while step < max_steps:
                # AI selects action
                action_idx = agent.select_action(state, game.get_valid_actions(game.pacman.position))
                action = ACTION_TO_DIRECTION[action_idx]
                
                _, reward, done, info = game.step(action)
                game_state = game.get_state()
                state = vision.extract_features(game_state, self.game_map)
                
                total_reward += reward
                step += 1
                
                if done:
                    break
            
            final_state = game.get_state()
            scores.append(final_state['score'])
            rewards.append(total_reward)
            steps_list.append(step)
            
            if final_state['game_state'] == 'won':
                wins += 1
            
            if (i + 1) % 5 == 0:
                print(f"Game {i+1}/{num_games}: Score={final_state['score']}, Steps={step}, "
                      f"Status={'WIN' if final_state['game_state'] == 'won' else 'LOST'}")
        
        results = {
            'agent': 'Neural Network',
            'scores': scores,
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'avg_reward': np.mean(rewards),
            'avg_steps': np.mean(steps_list),
            'wins': wins,
            'win_rate': wins / num_games * 100
        }
        
        self._print_results(results)
        return results
    
    def _print_results(self, results: Dict):
        """Print formatted results"""
        print(f"\n{results['agent']} Results:")
        print(f"  Average Score: {results['avg_score']:.1f} ± {results['std_score']:.1f}")
        print(f"  Score Range: [{results['min_score']}, {results['max_score']}]")
        print(f"  Average Reward: {results['avg_reward']:.1f}")
        print(f"  Average Steps: {results['avg_steps']:.1f}")
        print(f"  Win Rate: {results['win_rate']:.1f}% ({results['wins']}/{len(results['scores'])})")
    
    def compare(self, ai_model_path: str, num_games: int = 20):
        """Compare expert vs AI"""
        print("\n" + "=" * 60)
        print("AGENT COMPARISON")
        print("=" * 60)
        print(f"Difficulty: {self.config.get('ghost_ai_types')}")
        print(f"Games per agent: {num_games}")
        
        expert_results = self.evaluate_expert(num_games)
        ai_results = self.evaluate_ai(ai_model_path, num_games)
        
        # Comparison summary
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        
        metrics = [
            ('Average Score', 'avg_score', ':.1f'),
            ('Max Score', 'max_score', ''),
            ('Win Rate', 'win_rate', ':.1f'),
            ('Average Steps', 'avg_steps', ':.1f')
        ]
        
        print(f"{'Metric':<20} {'Expert':<15} {'AI':<15} {'Winner':<15}")
        print("-" * 65)
        
        for metric_name, metric_key, fmt in metrics:
            expert_val = expert_results[metric_key]
            ai_val = ai_results[metric_key]
            
            if fmt == ':.1f':
                expert_str = f"{expert_val:.1f}"
                ai_str = f"{ai_val:.1f}"
            elif fmt == ':':
                expert_str = f"{expert_val}"
                ai_str = f"{ai_val}"
            else:
                expert_str = str(int(expert_val))
                ai_str = str(int(ai_val))
            
            winner = "Expert" if expert_val > ai_val else "AI" if ai_val > expert_val else "Tie"
            print(f"{metric_name:<20} {expert_str:<15} {ai_str:<15} {winner:<15}")
        
        # Save results
        comparison = {
            'expert': expert_results,
            'ai': ai_results,
            'config': self.config
        }
        
        with open('comparison_results.json', 'w') as f:
            # Convert numpy types to native Python types for JSON
            def convert(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(item) for item in obj]
                return obj
            
            json.dump(convert(comparison), f, indent=2)
        
        print("\nResults saved to comparison_results.json")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Expert vs AI')
    parser.add_argument('--model', type=str, default='final_model.pth', help='AI model checkpoint')
    parser.add_argument('--games', type=int, default=20, help='Games per agent')
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'], default='medium',
                       help='Ghost difficulty')
    
    args = parser.parse_args()
    
    # Ghost AI configuration
    if args.difficulty == 'easy':
        ghost_ai_types = ['random', 'random', 'random', 'random']
    elif args.difficulty == 'medium':
        ghost_ai_types = ['random', 'random', 'chase', 'random']
    else:  # hard
        ghost_ai_types = ['chase', 'chase', 'chase', 'chase']
    
    config = {
        'ghost_ai_types': ghost_ai_types,
        'difficulty': args.difficulty
    }
    
    comparator = AgentComparator(config)
    comparator.compare(args.model, args.games)


if __name__ == '__main__':
    main()
