import random
from copy import deepcopy
import csv
import numpy as np
from tqdm import tqdm


class GamePiece:
    def __init__(self, size, player):
        self.size = size
        self.player = player

    def __str__(self):
        return f"P{self.player}S{self.size}"

    def __repr__(self):
        return self.__str__()

class GobbletGame:
    def __init__(self):
        # Make board and player stacks
        self.board = [[[] for _ in range(4)] for _ in range(4)]
        self.players = {1: [], 2: []}
        self.current_player = 1
        self.setup_pieces()

    def setup_pieces(self):
        for player in [1, 2]:
            self.players[player] = [
                [GamePiece(size, player) for size in range(4, 0, -1)]
                for _ in range(3)
            ]

    def get_game_state(self):
        return deepcopy(self)

    def simulate_move(self, move) -> "GobbletGame":
        new_game = deepcopy(self)
        new_game.make_move(move)
        return new_game

    def make_move(self, move):
        source_type, source_idx, target_x, target_y = move
        # Two types of moves
        if source_type == "stack":
            piece = self.players[self.current_player][source_idx].pop(0)
        else:
            source_x, source_y = source_idx
            piece = self.board[source_x][source_y].pop()

        if self.board[target_x][target_y]:
            top_piece = self.board[target_x][target_y][-1]
            if top_piece.size >= piece.size:
                if source_type == "stack":
                    self.players[self.current_player][source_idx].insert(0, piece)
                else:
                    self.board[source_x][source_y].append(piece)
                return False

        self.board[target_x][target_y].append(piece)

        winner = self.check_winner()
        if not winner:
            self.current_player = 3 - self.current_player

        return True

    def get_board_moves(self):
        moves = []
        for source_x in range(4):
            for source_y in range(4):
                if self.board[source_x][source_y]:
                    top_piece = self.board[source_x][source_y][-1]
                    # Make sure piece belongs to current player
                    if top_piece.player == self.current_player:
                        for target_x in range(4):
                            for target_y in range(4):
                                if source_x == target_x and source_y == target_y:
                                    continue
                                if not self.board[target_x][target_y] or \
                                        self.board[target_x][target_y][-1].size < top_piece.size:
                                    moves.append(("board", (source_x, source_y), target_x, target_y))
        return moves

    def get_stack_moves(self):
        moves = []
        for stack_idx, stack in enumerate(self.players[self.current_player]):
            if stack:
                piece = stack[0]
                for x in range(4):
                    for y in range(4):
                        if not self.board[x][y] or self.board[x][y][-1].size < piece.size:
                            moves.append(("stack", stack_idx, x, y))
        return moves

    def get_legal_moves(self):
        board_moves = self.get_board_moves()
        stack_moves = self.get_stack_moves()
        return board_moves + stack_moves

    def check_winner(self):
        # Check rows, columns, both diagonals to see if 4 in a row has been reached for a win
        for row in range(4):
            if all(self.board[row][col] for col in range(4)):
                if all(self.board[row][0][-1].player == self.board[row][col][-1].player for col in range(4)):
                    return self.board[row][0][-1].player

        for col in range(4):
            if all(self.board[row][col] for row in range(4)):
                if all(self.board[0][col][-1].player == self.board[row][col][-1].player for row in range(4)):
                    return self.board[0][col][-1].player

        if all(self.board[i][i] for i in range(4)):
            if all(self.board[0][0][-1].player == self.board[i][i][-1].player for i in range(4)):
                return self.board[0][0][-1].player

        if all(self.board[i][3 - i] for i in range(4)):
            if all(self.board[0][3][-1].player == self.board[i][3 - i][-1].player for i in range(4)):
                return self.board[0][3][-1].player

        return None

    def get_board_display(self):
        board_display = "    0   1   2   3\n\n"

        board_display += "  ┌"
        for col in range(3):
            board_display += "───┬"
        board_display += "───┐\n"

        for row in range(4):
            board_display += f"{row} │"
            for col in range(4):
                if not self.board[row][col]:
                    board_display += "   │"
                else:
                    board_display += f" {self.board[row][col][-1]} │"
            board_display += "\n"
            if row < 3:
                board_display += "  ├"
                for col in range(3):
                    board_display += "───┼"
                board_display += "───┤\n"
            else:
                board_display += "  └"
                for col in range(3):
                    board_display += "───┴"
                board_display += "───┘\n"

        return board_display

    def display_stacks(self, player):
        result = f"Player {player} Stacks:\n"
        for stack_idx, stack in enumerate(self.players[player]):
            result += f"Stack {stack_idx + 1}: "
            if not stack:
                result += "Empty"
            else:
                result += " ".join(str(piece) for piece in stack)
            result += "\n"
        return result

class RandomBot:
    def __init__(self, player_id):
        self.player_id = player_id

    def select_move(self, game_state):
        legal_moves = game_state.get_legal_moves()
        if legal_moves:
            return random.choice(legal_moves)
        return None

class HueristicMDPBot:
    def __init__(self, player_id):
        self.player_id = player_id

    def get_rewards(self, game:GobbletGame):
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None
        move_rewards = {}
        for move in legal_moves:
            move_rewards[move] = self.reward(game, move)
        return move_rewards

    def select_move(self, game:GobbletGame):
        move_to_reward = self.get_rewards(game)
        best_move = max(move_to_reward, key=move_to_reward.get)
        return best_move

    def reward(self, game, move) -> float:
        new_game = game.simulate_move(move)
        # First check if the move causes insta win or insta loss
        if new_game.check_winner() == self.player_id:
            return 1_000_000
        if new_game.check_winner() == 3 - self.player_id:
            return -1_000_000

        score = 0
        for line in self._get_all_lines(new_game.get_game_state()):
            for board_stack in line:
                if board_stack:
                    piece = board_stack[0]
                    score += piece.size * piece.player
        score *= -.25 if game.current_player == 2 else .25
        score += self._count_blocked_threats(game, new_game, self.player_id)
        score += self._count_threats(new_game, self.player_id)
        return score

    def _get_all_lines(self, game_state):
        board = game_state.board
        lines = []

        # Rows
        for r in range(4):
            lines.append([board[r][c] for c in range(4)])
        # Cols
        for c in range(4):
            lines.append([board[r][c] for r in range(4)])
        # Diagonals
        lines.append([board[i][i] for i in range(4)])
        lines.append([board[i][3 - i] for i in range(4)])
        return lines

    def _count_blocked_threats(self, old_game, new_game, player):
        opponent = 3 - player
        old_threats = self._count_threats(old_game, opponent)
        new_threats = self._count_threats(new_game, opponent)
        if new_threats < old_threats:
            return (old_threats - new_threats) * 0.5
        return 0.0

    def _count_threats(self, game_state, player):
        threat_val = 0.0
        lines = self._get_all_lines(game_state)
        for line in lines:
            top_players = [cell[-1].player for cell in line if cell]
            count_me = top_players.count(player)
            if count_me == 2:
                threat_val += 0.5
            elif count_me == 3:
                threat_val += 2
        return threat_val


def simulate_games(bot1_class, bot2_class, num_games=10, verbose=True):
    wins = {1: 0, 2: 0}
    for game_idx in range(num_games):
        game = GobbletGame()
        bot1 = bot1_class(1)
        bot2 = bot2_class(2)

        if verbose:
            print(f"\n--- Starting game {game_idx + 1} ---")

        while True:
            winner = game.check_winner()
            if winner:
                wins[winner] += 1
                if verbose:
                    print(f"Game {game_idx + 1} - Player {winner} wins!")
                break

            current_bot = bot1 if game.current_player == 1 else bot2
            move = current_bot.select_move(game.get_game_state())
            if not move:
                if verbose:
                    print("No legal moves available.")
                break
            success = game.make_move(move)

            if verbose and success:
                print(f"Player {game.current_player} made move {move}")
                print(game.get_board_display())

    if verbose:
        print("\nSimulation complete.")
        print(f"Total Wins: Player1: {wins[1]}, Player2: {wins[2]}")
    return wins

def cell_to_vector(cell):
    vec = [0,0,0,0]
    for idx, piece in enumerate(cell):
        mult = 1 if piece.player == 1 else -1
        vec[idx] = mult * piece.size
    return vec

def stack_to_vector(stack):
    vec = [0,0,0,0]
    for idx, piece in enumerate(stack):
        mult = 1 if piece.player == 1 else -1
        vec[idx] = mult * piece.size
    return vec

def vectorize_state(game):
    state_vec = []

    for stack in game.players[1]:
        stack_vec = stack_to_vector(stack)
        state_vec.append(stack_vec)
    for stack in game.players[2]:
        stack_vec = stack_to_vector(stack)
        state_vec.append(stack_vec)

    for row in range(4):
        for col in range(4):
            cell_vec = cell_to_vector(game.board[row][col])
            state_vec.append(cell_vec)

    return state_vec

def vectorize_move(move):
    from_type, from_num, x, y = move
    if from_type == "stack":
        return [1, from_num, -1, x, y]
    else:
        from_x, from_y = from_num
        return [0, from_x, from_y, x, y]

def simulate_random_vs_random(num_games=10, save_data=False, output_file="game_data.csv"):
    all_data = []

    for g in range(num_games):
        game = GobbletGame()
        bot1 = RandomBot(1)
        bot2 = RandomBot(2)
        outcome = None
        all_data_game = []

        while True:
            current_player = game.current_player
            state_vec = [[1 if current_player == 1 else -1]] + vectorize_state(game)

            move = bot1.select_move(game.get_game_state()) if current_player == 1 else bot2.select_move(game.get_game_state())

            if move:
                game.make_move(move)
            else:
                break

            move_vec = vectorize_move(move)
            winner = game.check_winner()

            if winner:
                outcome = 1 if winner == 1 else -1
                break

            all_data_game.append((state_vec, move_vec))

        for state_information in all_data_game:
            all_data.append((state_information[0], state_information[1], outcome))


    if save_data:
        with open(output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            for (s_vec, m_vec, res) in all_data:
                row = s_vec + m_vec + [res]
                writer.writerow(row)

    return all_data

def simulate_heuristic_vs_heuristic(num_games=10, save_data=False, output_file="game_data.csv"):
    all_data = []

    for g in range(num_games):
        game = GobbletGame()
        bot1 = HueristicMDPBot(1)
        bot2 = HueristicMDPBot(2)
        outcome = None
        all_data_game = []

        while True:
            current_player = game.current_player
            state_vec = [[1 if current_player == 1 else -1]] + vectorize_state(game)

            move = bot1.select_move(game.get_game_state()) if current_player == 1 else bot2.select_move(game.get_game_state())

            if move:
                game.make_move(move)
            else:
                break

            move_vec = vectorize_move(move)
            winner = game.check_winner()

            if winner:
                outcome = 1 if winner == 1 else -1
                break

            all_data_game.append((state_vec, move_vec))

        for state_information in all_data_game:
            all_data.append((state_information[0], state_information[1], outcome))


    if save_data:
        with open(output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            for (s_vec, m_vec, res) in all_data:
                row = s_vec + m_vec + [res]
                writer.writerow(row)

    return all_data

def simulate_random_vs_hueristic(num_games=10, save_data=False, output_file="game_data.csv"):
    all_data = []

    hueristic_wins = 0

    for g in tqdm(range(num_games)):
        game = GobbletGame()
        bot1 = HueristicMDPBot(1)
        bot2 = RandomBot(2)
        outcome = None
        all_data_game = []

        while True:
            current_player = game.current_player
            state_vec = [[1 if current_player == 1 else -1]] + vectorize_state(game)

            move = bot1.select_move(game.get_game_state()) if current_player == 1 else bot2.select_move(game.get_game_state())

            if move:
                game.make_move(move)
            else:
                break

            move_vec = vectorize_move(move)
            winner = game.check_winner()

            if winner:
                outcome = 1 if winner != current_player else -1
                if winner == 1:
                    hueristic_wins += 1
                break

            all_data_game.append((state_vec, move_vec))

        for state_information in all_data_game:
            all_data.append((state_information[0], state_information[1], outcome))


    if save_data:
        with open(output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            for (s_vec, m_vec, res) in all_data:
                row = s_vec + m_vec + [res]
                writer.writerow(row)

    print(f"Hueristic Wins: {hueristic_wins}")
    print(f"Total Games: {num_games}")

    return all_data

if __name__ == "__main__":
    # simulate_ppo_vs_ppo(save_data=True)
    # simulate_ppo_vs_ppo(save_data=True)
    simulate_random_vs_hueristic(save_data=True)
    # game = GobbletGame()
    # bot1 = PPOBot(1)
    # bot2 = PPOBot(2)
    # print(bot1.get_rewards(game=game))

