import math
import random
from copy import deepcopy
from gobblet_game import GobbletGame, RandomBot, HueristicMDPBot
from ppo import PPOBot
from tqdm import tqdm

class Node:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move  # The move that led to this state
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = game_state.get_legal_moves()
        self.value = 0.0  # Add value field for heuristic evaluation
        
    def add_child(self, move, game_state):
        """Add a child node for the given move and game state"""
        child = Node(game_state, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child
        
    def update(self, result):
        """Update node statistics"""
        self.visits += 1
        self.wins += result
        
    def fully_expanded(self):
        """Check if all possible moves have been tried"""
        return len(self.untried_moves) == 0
        
    def best_child(self, c_param=1.414):
        """Select the best child using UCT formula with heuristic value"""
        if not self.children:
            return None
            
        choices = []
        for c in self.children:
            if c.visits == 0:
                return c  # Return unvisited child immediately
                
            uct_value = (c.wins / c.visits) + c_param * math.sqrt((2 * math.log(self.visits) / c.visits))
            heuristic_value = c.value / 1000  # Normalize heuristic value
            combined_value = 0.7 * uct_value + 0.3 * heuristic_value
            choices.append(combined_value)
            
        return self.children[choices.index(max(choices))]
        
    def tree_string(self, indent=''):
        """String representation of the tree for debugging"""
        s = indent + str(self)
        for c in self.children:
            s += '\n' + c.tree_string(indent + '  ')
        return s
        
    def __str__(self):
        """String representation of the node"""
        return f"Node(wins={self.wins}, visits={self.visits}, value={self.value:.2f}, moves={len(self.untried_moves)})"

class MCTS:
    def __init__(self, player_id, exploration_weight=1.414, max_depth=5):
        self.player_id = player_id
        self.exploration_weight = exploration_weight
        self.heuristic_bot = HueristicMDPBot(player_id)
        self.opponent_bot = HueristicMDPBot(3 - player_id)  # Create opponent bot
        self.random_bot = RandomBot(3 - player_id)  # Create random bot for opponent
        self.max_depth = max_depth
        
    def _evaluate_state(self, game_state):
        """Use HeuristicMDPBot's reward function directly"""
        # Get all legal moves in this state
        legal_moves = game_state.get_legal_moves()
        if not legal_moves:
            return 0
            
        # Use the heuristic bot's reward function to evaluate the best move
        move_rewards = {move: self.heuristic_bot.reward(game_state, move) 
                       for move in legal_moves}
        return max(move_rewards.values())  # Return the best possible reward
        
    def _simulate(self, game_state):
        """Use hybrid rollout policy - heuristic for MCTS player, mixed for opponent"""
        state = deepcopy(game_state)
        depth = 0
        max_sim_depth = 5
        
        while not state.check_winner() and depth < max_sim_depth:
            if state.current_player == self.player_id:
                # MCTS player uses pure heuristic strategy
                move_rewards = self.heuristic_bot.get_rewards(state)
                if not move_rewards:
                    break
                best_move = max(move_rewards.items(), key=lambda x: x[1])[0]
            else:
                # Opponent uses mixed strategy (70% heuristic, 30% random)
                if random.random() < 0.7:
                    move_rewards = self.opponent_bot.get_rewards(state)
                    if not move_rewards:
                        break
                    best_move = max(move_rewards.items(), key=lambda x: x[1])[0]
                else:
                    legal_moves = state.get_legal_moves()
                    if not legal_moves:
                        break
                    best_move = random.choice(legal_moves)
            
            state.make_move(best_move)
            depth += 1
            
        # If we reached a terminal state, use the actual winner
        winner = state.check_winner()
        if winner:
            return 1.0 if winner == self.player_id else 0.0
            
        # If we hit depth limit, use heuristic evaluation
        final_value = self._evaluate_state(state)
        return 1.0 if final_value > 0 else (0.0 if final_value < 0 else 0.5)
        
    def choose_move(self, game_state, num_simulations=1000):
        """Choose the best move using MCTS with heuristic simulation"""
        root = Node(game_state)
        root.value = self._evaluate_state(game_state)
        
        # Early stopping if we have a clear winning move
        if root.value >= 1_000_000:
            return game_state.get_legal_moves()[0]
            
        for _ in tqdm(range(num_simulations), desc="Simulating MCTS"):
            node = root
            state = deepcopy(game_state)
            depth = 0
            
            # Selection with depth limit
            while node.fully_expanded() and node.children and depth < self.max_depth:
                node = node.best_child(self.exploration_weight)
                if node is None:
                    break
                state.make_move(node.move)
                depth += 1
            
            # Expansion if within depth limit
            if node.untried_moves and depth < self.max_depth:
                # Use heuristic bot to select the best untried move
                move_rewards = {move: self.heuristic_bot.reward(state, move) 
                              for move in node.untried_moves}
                move = max(move_rewards.items(), key=lambda x: x[1])[0]
                state.make_move(move)
                node = node.add_child(move, deepcopy(state))
                node.value = self._evaluate_state(state)
            
            # Simulation using heuristic bot
            result = self._simulate(state)
            
            # Backpropagation
            while node is not None:
                node.update(result)
                node = node.parent
                
        # Choose the move with the highest combined score
        if not root.children:
            return None
            
        best_child = max(root.children, key=lambda c: (c.wins / c.visits) + (c.value / 1000))
        return best_child.move

class MCTSBot:
    def __init__(self, player_id, num_simulations=1000):  # Reduced from 1000
        self.player_id = player_id
        self.mcts = MCTS(player_id)
        self.num_simulations = num_simulations
        
    def select_move(self, game_state):
        """Select a move using MCTS"""
        return self.mcts.choose_move(game_state, self.num_simulations)

def simulate_mcts_vs_random(num_games=10, mcts_simulations=1000):  # Reduced from 1000
    """Simulate MCTS bot vs Random bot"""
    wins = {1: 0, 2: 0}
    
    for game_idx in range(num_games):
        game = GobbletGame()
        mcts_bot = MCTSBot(1, mcts_simulations)
        random_bot = RandomBot(2)
        
        print(f"\nStarting game {game_idx + 1}")
        
        while True:
            current_player = game.current_player
            current_bot = mcts_bot if current_player == 1 else random_bot
            
            move = current_bot.select_move(game.get_game_state())
            if not move:
                break
                
            game.make_move(move)
            print(game.get_board_display())
            
            winner = game.check_winner()
            if winner:
                wins[winner] += 1
                print(f"Game {game_idx + 1} - Player {winner} wins!")
                break
                
    print("\nSimulation complete!")
    print(f"Total wins - MCTS: {wins[1]}, Random: {wins[2]}")
    return wins

def simulate_mcts_vs_heuristic(num_games=10, mcts_simulations=1000):  # Reduced from 1000
    """Simulate MCTS bot vs Random bot"""
    wins = {1: 0, 2: 0}
    
    for game_idx in range(num_games):
        game = GobbletGame()
        mcts_bot = MCTSBot(1, mcts_simulations)
        random_bot = HueristicMDPBot(2)
        
        print(f"\nStarting game {game_idx + 1}")
        
        while True:
            current_player = game.current_player
            current_bot = mcts_bot if current_player == 1 else random_bot
            
            move = current_bot.select_move(game.get_game_state())
            if not move:
                break
                
            game.make_move(move)
            print(game.get_board_display())
            
            winner = game.check_winner()
            if winner:
                wins[winner] += 1
                print(f"Game {game_idx + 1} - Player {winner} wins!")
                break
                
    print("\nSimulation complete!")
    print(f"Total wins - MCTS: {wins[1]}, Random: {wins[2]}")
    return wins

def simulate_mcts_vs_ppo(num_games=10, mcts_simulations=1000):  # Reduced from 1000
    """Simulate MCTS bot vs PPO bot"""
    wins = {1: 0, 2: 0}
    
    for game_idx in range(num_games):
        game = GobbletGame()
        mcts_bot = MCTSBot(1, mcts_simulations)
        ppo_bot = PPOBot(2)
        ppo_bot.load_model("final_model.pt")
        
        print(f"\nStarting game {game_idx + 1}")
        
        while True:
            current_player = game.current_player
            current_bot = mcts_bot if current_player == 1 else ppo_bot

            move = current_bot.select_move(game.get_game_state())
            if not move:
                break
                
            game.make_move(move)
            print(game.get_board_display())

            winner = game.check_winner()
            if winner:
                wins[winner] += 1
                print(f"Game {game_idx + 1} - Player {winner} wins!")
                break
                
    print("\nSimulation complete!")
    print(f"Total wins - MCTS: {wins[1]}, PPO: {wins[2]}")
    return wins


if __name__ == "__main__":
    # Test MCTS against random bot
    #simulate_mcts_vs_random(num_games=10, mcts_simulations=50) 
    # Test MCTS against heuristic bot
    #simulate_mcts_vs_heuristic(num_games=10, mcts_simulations=5)
    # Test MCTS against PPO bot
    simulate_mcts_vs_ppo(num_games=10, mcts_simulations=1000)