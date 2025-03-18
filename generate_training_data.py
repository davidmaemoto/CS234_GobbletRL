from gobblet_game import GobbletGame, RandomBot, vectorize_state, vectorize_move, HueristicMDPBot
import csv
from tqdm import tqdm
import numpy as np

def compute_reward(old_state, new_state, action, player_id):
    r = 0.0
    
    winner = new_state.check_winner()
    if winner:
        return 1_000_000 if winner == player_id else -1_000_000
        
    opponent_id = 3 - player_id
    opponent_can_win = False
    for move in new_state.get_legal_moves():
        simulated_state = new_state.simulate_move(move)
        if simulated_state.check_winner() == opponent_id:
            opponent_can_win = True
            break
    if opponent_can_win:
        r -= 1_000_000

    score = 0
    for line in _get_all_lines(new_state):
        for board_stack in line:
            if board_stack:
                piece = board_stack[-1]
                score += piece.size * piece.player
    score *= -.25 if new_state.current_player == 2 else .25
    r += score
    
    r += _count_threats(new_state, player_id)
    r += _count_blocked_threats(old_state, new_state, player_id)
    r -= 0.8 * _count_threats(new_state, opponent_id)
    r -= 0.01
    
    return r

def _get_all_lines(game_state):
    board = game_state.board
    lines = []
    for r in range(4):
        lines.append([board[r][c] for c in range(4)])
    for c in range(4):
        lines.append([board[r][c] for r in range(4)])
    lines.append([board[i][i] for i in range(4)])
    lines.append([board[i][3-i] for i in range(4)])
    return lines

def _count_threats(game_state, player):
    t = 0.0
    
    lines = _get_all_lines(game_state)
    for line in lines:
        top_players = [cell[-1].player for cell in line if cell]
        count_me = top_players.count(player)
        if count_me == 2:
            t += 0.5
        elif count_me == 3:
            t += 2.0
            
    return t
    
def _count_blocked_threats(old_game, new_game, player):
    opponent = 3 - player
    old = _count_threats(old_game, opponent)
    new = _count_threats(new_game, opponent)
    diff = old - new
    if new < old:
        return diff * 0.5
    return 0.0

def vectorize_move(move):
    move_type, from_idx, x, y = move
    if move_type == "board":
        from_x, from_y = from_idx
        return [0, from_x, from_y, x, y]
    else:
        stack_num = from_idx
        return [1, stack_num, -1, x, y]  # stack move

def generate_random_games(num_games=1000, output_file="game_data2.csv"):
    all_data = []

    bot1 = RandomBot(1)
    bot2 = RandomBot(2)
    
    for _ in tqdm(range(num_games)):
        game = GobbletGame()
        while True:
            current_player = game.current_player
            current_bot = bot1 if current_player == 1 else bot2
            current_state = game.get_game_state()
            state_vec = [1 if current_player == 1 else -1] + [x for vec in vectorize_state(current_state) for x in (vec if isinstance(vec, list) else [vec])]

            move = current_bot.select_move(current_state)
            if not move:
                break
            old_state = game.get_game_state()
            made_move = game.make_move(move)
            
            if made_move:
                new_state = game.get_game_state()
                reward = compute_reward(old_state, new_state, move, current_player)
                next_state_vec = [1 if game.current_player == 1 else -1] + [x for vec in vectorize_state(new_state) for x in (vec if isinstance(vec, list) else [vec])]
                
                move_vec = vectorize_move(move)
                if move_vec is None:
                    continue
                row = state_vec + move_vec + [reward] + next_state_vec
                all_data.append(row)

                winner = game.check_winner()
                if winner:
                    final_reward = 1_000_000 if winner == current_player else -1_000_000
                    # final_reward = 10.0 if winner == current_player else -10.0
                    all_data[-1][len(state_vec) + len(move_vec)] = final_reward
                    break
                
                #if len(all_data) > 100:
                #    all_data[-1][93] = -0.1
                #    break
    
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_data)

if __name__ == "__main__":
    generate_random_games(num_games=100000, output_file="game_data2.csv")