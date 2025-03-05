from gobblet_game import GobbletGame, RandomBot, vectorize_state, vectorize_move, HueristicMDPBot
import csv
from tqdm import tqdm
import numpy as np

def compute_reward(old_state, new_state, action, player_id):
    """Compute immediate reward for a state transition"""
    reward = 0.0
    
    # Win/loss rewards (highest priority)
    winner = new_state.check_winner()
    if winner:
        return 1_000_000 if winner == player_id else -1_000_000
        
    # Check if opponent can win in one move (highest priority after win/loss)
    opponent_id = 3 - player_id
    opponent_can_win = False
    for move in new_state.get_legal_moves():
        simulated_state = new_state.simulate_move(move)
        if simulated_state.check_winner() == opponent_id:
            opponent_can_win = True
            break
    if opponent_can_win:
        reward -= 1_000_000  # Heavy penalty for allowing opponent winning move
        
    # Piece positioning score
    score = 0
    for line in _get_all_lines(new_state):
        for board_stack in line:
            if board_stack:
                piece = board_stack[-1]  # Get top piece
                score += piece.size * piece.player
    score *= -.25 if new_state.current_player == 2 else .25
    reward += score
    
    # Threat-based rewards
    reward += _count_threats(new_state, player_id)
    reward += _count_blocked_threats(old_state, new_state, player_id)
    reward -= 0.8 * _count_threats(new_state, opponent_id)
    
    # Add small penalty for moves that don't progress the game
    reward -= 0.01
    
    return reward

def _get_all_lines(game_state):
    """Get all possible lines (rows, columns, diagonals)"""
    board = game_state.board
    lines = []
    
    # Rows
    for r in range(4):
        lines.append([board[r][c] for c in range(4)])
        
    # Columns
    for c in range(4):
        lines.append([board[r][c] for r in range(4)])
        
    # Diagonals
    lines.append([board[i][i] for i in range(4)])
    lines.append([board[i][3-i] for i in range(4)])
    
    return lines

def _count_threats(game_state, player):
    """Count the number of threatening positions"""
    threat_value = 0.0
    
    lines = _get_all_lines(game_state)
    for line in lines:
        top_players = [cell[-1].player for cell in line if cell]
        count_me = top_players.count(player)
        if count_me == 2:
            threat_value += 0.5
        elif count_me == 3:
            threat_value += 2.0
            
    return threat_value
    
def _count_blocked_threats(old_game, new_game, player):
    """Compute reward for blocking opponent threats"""
    opponent = 3 - player
    old_threats = _count_threats(old_game, opponent)
    new_threats = _count_threats(new_game, opponent)
    
    if new_threats < old_threats:
        return (old_threats - new_threats) * 0.5
    return 0.0

def vectorize_move(move):
    """Convert a move to a vector format with [source_type, source_x, source_y, target_x, target_y]
    source_type: 0 for board, 1 for stack
    source_x, source_y: coordinates for board moves, (stack_idx, -1) for stack moves
    target_x, target_y: target coordinates
    """
    source_type, source_idx, tx, ty = move
    if source_type == "board":
        sx, sy = source_idx
        return [0, sx, sy, tx, ty]  # board move
    else:  # stack move
        stack_num = source_idx
        return [1, stack_num, -1, tx, ty]  # stack move

def generate_random_games(num_games=1000, output_file="game_data2.csv"):
    """Generate training data from random bot vs random bot games"""
    print(f"Generating {num_games} games...")
    
    # List to store all game data
    all_data = []
    
    # Create bots
    bot1 = RandomBot(1)  # Use Random bot for diverse training data
    bot2 = RandomBot(2)  # Keep random bot as opponent for diversity
    
    # Generate games with progress bar
    for _ in tqdm(range(num_games)):
        game = GobbletGame()
        
        while True:
            current_player = game.current_player
            current_bot = bot1 if current_player == 1 else bot2
            
            # Get current state
            current_state = game.get_game_state()
            # Combine current player and state vector properly
            state_vec = [1 if current_player == 1 else -1] + [x for vec in vectorize_state(current_state) for x in (vec if isinstance(vec, list) else [vec])]
            
            # Get bot's move
            move = current_bot.select_move(current_state)
            if not move:
                break
                
            # Make the move and get next state
            old_state = game.get_game_state()
            success = game.make_move(move)
            
            if success:
                new_state = game.get_game_state()
                
                # Get immediate reward with improved calculation
                reward = compute_reward(old_state, new_state, move, current_player)
                
                # Get next state vector
                next_state_vec = [1 if game.current_player == 1 else -1] + [x for vec in vectorize_state(new_state) for x in (vec if isinstance(vec, list) else [vec])]
                
                # Convert move to vector format
                move_vec = vectorize_move(move)
                if move_vec is None:
                    continue  # Skip invalid moves
                
                # Store transition
                row = state_vec + move_vec + [reward] + next_state_vec
                all_data.append(row)
                
                # Check if game is over
                winner = game.check_winner()
                if winner:
                    # Update final rewards for the winning sequence
                    final_reward = 1_000_000 if winner == current_player else -1_000_000
                    # final_reward = 10.0 if winner == current_player else -10.0
                    all_data[-1][len(state_vec) + len(move_vec)] = final_reward  # Update the reward in the last transition
                    break
                
                # Add small penalty for very long games
                #if len(all_data) > 100:  # If game is too long
                #    all_data[-1][93] = -0.1  # Small penalty for the last move
                #    break
    
    # Save to CSV
    print(f"\nSaving {len(all_data)} transitions to {output_file}...")
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_data)
    
    print("Data generation complete!")
    print(f"Total state transitions saved: {len(all_data)}")

if __name__ == "__main__":
    # Generate training data
    generate_random_games(num_games=100000, output_file="game_data2.csv") 