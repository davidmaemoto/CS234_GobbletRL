import math
import random
from copy import deepcopy
from gobblet_game import GobbletGame, RandomBot, HueristicMDPBot
from ppo import PPOBot
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from collections import defaultdict
import pandas as pd

class Node:
    def __init__(self, parent=None, move=None):
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = None
        self.value = 0.0
        
    def add_child(self, move, game_state):
        child = Node(parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child
        
    def update_node_values(self, result):
        self.visits += 1
        self.wins += result
        
    def tried_all(self):
        return len(self.untried_moves) == 0
        
    def best_child(self, c_param=1.414):
        if not self.children:
            return None
            
        choices = []
        for c in self.children:
            if c.visits == 0:
                return c
                
            uct_value = (c.wins / c.visits) + c_param * math.sqrt((2 * math.log(self.visits) / c.visits))
            value = 0.7 * uct_value + 0.3 * c.value/1000 # Divide by 1000 in order to normalize
            choices.append(value)
            
        return self.children[choices.index(max(choices))]
        
    def tree_string(self, indent=''):
        s = indent + str(self)
        for c in self.children:
            s += '\n' + c.tree_string(indent + '  ')
        return s
        
    def __str__(self):
        return f"Node(wins={self.wins}, visits={self.visits}, value={self.value:.2f}, moves={len(self.untried_moves)})"

class MCTS:
    def __init__(self, player_id, exploration_weight=1.414, max_depth=10):
        self.player_id = player_id
        self.exploration_weight = exploration_weight
        self.heuristic_bot = HueristicMDPBot(player_id)
        self.opponent_bot = HueristicMDPBot(3 - player_id)
        self.max_depth = max_depth

    def _evaluate_state(self, game_state):
        legal_moves = game_state.get_legal_moves()
        if not legal_moves:
            return 0
        move_rewards = {move: self.heuristic_bot.reward(game_state, move)
                       for move in legal_moves}
        return max(move_rewards.values())
        
    def choose_move(self, game_state, num_simulations=1000):
        root = Node()
        root.untried_moves = game_state.get_legal_moves()
        
        state_copy = deepcopy(game_state)
        
        for _ in range(num_simulations):
            state = deepcopy(state_copy)
            node = root
            
            while node.untried_moves is None or not node.untried_moves:
                if not node.children:
                    break
                node = node.best_child(self.exploration_weight)
                if node is None:
                    break
                state.make_move(node.move)
            
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                state.make_move(move)
                child = Node(parent=node, move=move)
                child.untried_moves = state.get_legal_moves()
                node.untried_moves.remove(move)
                node.children.append(child)
                node = child
            
            winner = state.check_winner()
            if winner:
                result = 1.0 if winner == self.player_id else 0.0
            else:
                result = self._quick_evaluate(state)
            
            while node is not None:
                node.update_node_values(result)
                node = node.parent
        
        return max(root.children, key=lambda c: c.visits).move

    def _quick_evaluate(self, state):
        if self._has_winning_threat(state, self.player_id):
            return 1.0
        if self._has_winning_threat(state, 3 - self.player_id):
            return -1.0
            
        score = 0.0
        
        center_positions = [(1,1), (1,2), (2,1), (2,2)]
        for x, y in center_positions:
            cell = state.board[x][y]
            if cell:
                top_piece = cell[-1]
                if top_piece.player == self.player_id:
                    score += 0.1
                else:
                    score -= 0.1
        
        for row in range(4):
            for col in range(4):
                cell = state.board[row][col]
                if cell:
                    top_piece = cell[-1]
                    if top_piece.player == self.player_id:
                        score += 0.05 * top_piece.size
                    else:
                        score -= 0.05 * top_piece.size
        
        return 0.5 + (score / 4.0)

    def _has_winning_threat(self, state, player):
        for row in state.board:
            count = 0
            for cell in row:
                if cell and cell[-1].player == player:
                    count += 1
            if count == 3:
                return True
                
        for col in range(4):
            count = 0
            for row in range(4):
                cell = state.board[row][col]
                if cell and cell[-1].player == player:
                    count += 1
            if count == 3:
                return True
                
        diag1 = [(0,0), (1,1), (2,2), (3,3)]
        diag2 = [(0,3), (1,2), (2,1), (3,0)]
        
        for diag in [diag1, diag2]:
            count = 0
            for x, y in diag:
                cell = state.board[x][y]
                if cell and cell[-1].player == player:
                    count += 1
            if count == 3:
                return True
                
        return False

    def _evaluate_position(self, state):
        return 1.0 if self._evaluate_state(state) > 0 else (0.0 if self._evaluate_state(state) < 0 else 0.5)

    def _rollout(self, state, current_depth):
        state = deepcopy(state)
        depth = current_depth
        
        while not state.check_winner() and depth < self.max_depth:
            legal_moves = state.get_legal_moves()
            if not legal_moves:
                break
            
            if random.random() < 0.7:
                move_rewards = self.heuristic_bot.get_rewards(state)
                move = max(move_rewards.items(), key=lambda x: x[1])
                move = move[0]
            else:
                move = random.choice(legal_moves)
                
            state.make_move(move)
            depth += 1
        
        winner = state.check_winner()
        if winner:
            return 1.0 if winner == self.player_id else 0.0
        
        return self._evaluate_position(state)

class MCTSBot:
    def __init__(self, player_id, num_simulations=1000):
        self.player_id = player_id
        self.mcts = MCTS(player_id)
        self.num_simulations = num_simulations
        
    def select_move(self, game_state):
        """Select a move using MCTS"""
        return self.mcts.choose_move(game_state, self.num_simulations)

def simulate_mcts_vs_random(num_games=10, mcts_simulations=1000):
    """Simulate MCTS bot vs Random bot"""
    wins = {1: 0, 2: 0}
    
    for game_idx in range(num_games):
        game = GobbletGame()
        mcts_bot = MCTSBot(1, mcts_simulations)
        random_bot = RandomBot(2)
        
        print(f"\nStarting g {game_idx + 1}")
        
        while True:
            current_player = game.current_player
            current_bot = mcts_bot if current_player == 1 else random_bot
            
            move = current_bot.select_move(game.get_game_state())
            if not move:
                break
                
            game.make_move(move)
            #print(game.get_board_display())
            
            winner = game.check_winner()
            if winner:
                wins[winner] += 1
                print(f"G {game_idx + 1} - P {winner} wins")
                break
                
    print(f"Wins: MCTS: {wins[1]}, Random: {wins[2]}")
    return wins

def simulate_mcts_vs_heuristic(num_games=10, mcts_simulations=1000):
    wins = {1: 0, 2: 0}
    
    for game_idx in range(num_games):
        game = GobbletGame()
        mcts_bot = MCTSBot(1, mcts_simulations)
        random_bot = HueristicMDPBot(2)
        
        print(f"\nG {game_idx + 1}")
        
        while True:
            current_player = game.current_player
            current_bot = mcts_bot if current_player == 1 else random_bot
            
            move = current_bot.select_move(game.get_game_state())
            if not move:
                break
                
            game.make_move(move)
            #print(game.get_board_display())
            
            winner = game.check_winner()
            if winner:
                wins[winner] += 1
                print(f"G {game_idx + 1} - P {winner}")
                break
                
    print(f"Wins: MCTS: {wins[1]}, Hueristic: {wins[2]}")
    return wins

def simulate_mcts_vs_ppo(num_games=10, mcts_simulations=1000):
    wins = {1: 0, 2: 0}
    
    for game_idx in range(num_games):
        game = GobbletGame()
        mcts_bot = MCTSBot(1, mcts_simulations)
        ppo_bot = PPOBot(player_id=2)
        ppo_bot.load_model("ppo_vs_random_final.pt")
        
        print(f"\nG {game_idx + 1}")
        
        while True:
            current_player = game.current_player
            current_bot = mcts_bot if current_player == 1 else ppo_bot

            move = current_bot.select_move(game.get_game_state()) if current_player == 1 else current_bot.collect_trajectory(game.get_game_state(), "ppo")
            if not move:
                break
                
            game.make_move(move)
            #print(game.get_board_display())

            winner = game.check_winner()
            if winner:
                wins[winner] += 1
                print(f"G {game_idx + 1} - P {winner} wins")
                break
                
    print(f"Wins: MCTS: {wins[1]}, PPO: {wins[2]}")
    return wins

def run_win_rate_experiment(sim_counts, games_per_count=100):
    results = {
        'PPO': [],
        'Hueristic': [],
        'Random': []
    }
    times = []
    
    for sims in sim_counts:
        print(f"\n{sims} number of Sims")
        move_times = []
        
        ppo_wins = 0
        game = GobbletGame()
        mcts_bot = MCTSBot(1, sims)
        ppo_bot = PPOBot(player_id=2)
        ppo_bot.load_model("ppo_vs_random_initial.pt")
        for _ in tqdm(range(games_per_count)):
            game = GobbletGame()
            while True:
                current_player = game.current_player
                if current_player == 1:
                    move_start = time.time()
                    move = mcts_bot.select_move(game.get_game_state())
                    move_times.append(time.time() - move_start)
                else:
                    move = ppo_bot.collect_trajectory(game.get_game_state(), "ppo")
                
                if not move:
                    break
                    
                game.make_move(move)
                winner = game.check_winner()
                if winner:
                    if winner == 1:
                        ppo_wins += 1
                    break
        
        results['PPO'].append(ppo_wins / games_per_count * 100)
        print(f"MCTS vs ppo win %: {results['PPO'][-1]:.1f}%")

        hueristic_wins = 0
        mcts_bot = MCTSBot(1, sims)
        hueristic_bot = HueristicMDPBot(2)
        
        for _ in tqdm(range(games_per_count)):
            game = GobbletGame()
            while True:
                current_player = game.current_player
                if current_player == 1:
                    move_start = time.time()
                    move = mcts_bot.select_move(game.get_game_state())
                    move_times.append(time.time() - move_start)
                else:
                    move = hueristic_bot.select_move(game.get_game_state())
                
                if not move:
                    break
                    
                game.make_move(move)
                winner = game.check_winner()
                if winner:
                    if winner == 1:
                        hueristic_wins += 1
                    break
        
        results['Hueristic'].append(hueristic_wins / games_per_count * 100)
        print(f"MCTS vs Heu win %: {results['Hueristic'][-1]:.1f}%")
        random_wins = 0
        mcts_bot = MCTSBot(1, sims)
        random_bot = RandomBot(2)
        
        for _ in tqdm(range(games_per_count)):
            game = GobbletGame()
            while True:
                current_player = game.current_player
                if current_player == 1:
                    move_start = time.time()
                    move = mcts_bot.select_move(game.get_game_state())
                    move_times.append(time.time() - move_start)
                else:
                    move = random_bot.select_move(game.get_game_state())
                
                if not move:
                    break
                    
                game.make_move(move)
                winner = game.check_winner()
                if winner:
                    if winner == 1:
                        random_wins += 1
                    break
        
        results['Random'].append(random_wins / games_per_count * 100)
        times.append(np.mean(move_times))
        
        print(f"Sim count: {sims}")
        print(f"Win % - vs ppo: {results['PPO'][-1]:.1f}%, vs Heu: {results['Hueristic'][-1]:.1f}%, vs Rand: {results['Random'][-1]:.1f}%")
        print(f"Avg t per move: {times[-1]:.3f} seconds")
    
    return sim_counts, results, times

def plot_win_rates(sim_counts, results):
    plt.figure(figsize=(10, 6))
    for opponent, win_rates in results.items():
        plt.plot(sim_counts, win_rates, marker='o', label=f'vs {opponent}')
    plt.xscale('log')
    plt.xlabel('Number of MCTS Simulations (log scale)')
    plt.ylabel('MCTS Win Rate (%)')
    plt.title('MCTS Performance vs Number of Simulations')
    plt.grid(True)
    plt.legend()
    plt.savefig('mcts_win_rates.png')
    plt.close()

def plot_computation_time(sim_counts, times):
    plt.figure(figsize=(10, 6))
    plt.scatter(sim_counts, times, alpha=0.6)
    plt.plot(sim_counts, times, '--', alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Average Time per Move (seconds, log scale)')
    plt.title('MCTS Computation Time vs Number of Simulations')
    plt.xlabel('Number of MCTS Simulations (log scale)')
    plt.grid(True)
    plt.savefig('mcts_computation_time.png')
    plt.close()

def create_all_visualizations():
    sim_counts = [10, 100, 1000, 10000]
    sim_counts, results, times = run_win_rate_experiment(sim_counts)
    plot_win_rates(sim_counts, results)
    plot_computation_time(sim_counts, times)
    

if __name__ == "__main__":
    #simulate_mcts_vs_random(num_games=100, mcts_simulations=5000)
    simulate_mcts_vs_heuristic(num_games=10, mcts_simulations=10000)
    #simulate_mcts_vs_ppo(num_games=100, mcts_simulations=5000)
    #create_all_visualizations()