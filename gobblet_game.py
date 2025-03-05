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
        # Initialize empty board (4x4 grid with empty stacks)
        self.board = [[[] for _ in range(4)] for _ in range(4)]
        # Initialize player pieces (3 stacks of 4 pieces per player)
        self.players = {1: [], 2: []}
        self.current_player = 1
        self.setup_pieces()

    def setup_pieces(self):
        """Initialize pieces for both players."""
        for player in [1, 2]:
            # Create 3 stacks, each with sizes 1-4 (smallest to largest)
            self.players[player] = [
                [GamePiece(size, player) for size in range(4, 0, -1)]
                for _ in range(3)
            ]

    def get_game_state(self):
        """Return a deep copy of the game state for agents to inspect."""
        return deepcopy(self)

    def simulate_move(self, move) -> "GobbletGame":
        new_game = deepcopy(self)
        new_game.make_move(move)
        return new_game

    def make_move(self, move):
        """Apply a move to the game state."""
        source_type, source_idx, target_x, target_y = move
        # Get the piece to move
        if source_type == "stack":
            # Remove piece from player's stack
            piece = self.players[self.current_player][source_idx].pop(0)
        else:  # source_type == "board"
            source_x, source_y = source_idx
            # Remove piece from board
            piece = self.board[source_x][source_y].pop()

        # Check if target location has a smaller piece
        if self.board[target_x][target_y]:
            top_piece = self.board[target_x][target_y][-1]
            if top_piece.size >= piece.size:
                # Invalid move, put the piece back
                if source_type == "stack":
                    self.players[self.current_player][source_idx].insert(0, piece)
                else:
                    self.board[source_x][source_y].append(piece)
                return False

        # Place the piece at the target location
        self.board[target_x][target_y].append(piece)

        # Check for winner
        winner = self.check_winner()
        if not winner:
            # Switch player only if no winner
            self.current_player = 3 - self.current_player

        return True

    def get_board_moves(self):
        """Get all possible moves from the board."""
        moves = []
        for source_x in range(4):
            for source_y in range(4):
                if self.board[source_x][source_y]:
                    top_piece = self.board[source_x][source_y][-1]

                    # Can only move player's own pieces
                    if top_piece.player == self.current_player:
                        for target_x in range(4):
                            for target_y in range(4):
                                # Skip same position
                                if source_x == target_x and source_y == target_y:
                                    continue

                                # Can place if empty or top piece is smaller
                                if not self.board[target_x][target_y] or \
                                        self.board[target_x][target_y][-1].size < top_piece.size:
                                    moves.append(("board", (source_x, source_y), target_x, target_y))
        return moves

    def get_stack_moves(self):
        """Get all possible moves from player's stacks."""
        moves = []
        for stack_idx, stack in enumerate(self.players[self.current_player]):
            if stack:  # If stack is not empty
                piece = stack[0]  # Top piece in the stack

                # Check all board positions
                for x in range(4):
                    for y in range(4):
                        # Can place if empty or top piece is smaller
                        if not self.board[x][y] or self.board[x][y][-1].size < piece.size:
                            moves.append(("stack", stack_idx, x, y))
        return moves

    def get_legal_moves(self):
        """Return a list of legal moves for the current player."""
        board_moves = self.get_board_moves()
        stack_moves = self.get_stack_moves()
        return board_moves + stack_moves

    def check_winner(self):
        """Check if there's a winner and return the player number or None."""
        # Check rows
        for row in range(4):
            if all(self.board[row][col] for col in range(4)):
                if all(self.board[row][0][-1].player == self.board[row][col][-1].player for col in range(4)):
                    return self.board[row][0][-1].player

        # Check columns
        for col in range(4):
            if all(self.board[row][col] for row in range(4)):
                if all(self.board[0][col][-1].player == self.board[row][col][-1].player for row in range(4)):
                    return self.board[0][col][-1].player

        # Check main diagonal
        if all(self.board[i][i] for i in range(4)):
            if all(self.board[0][0][-1].player == self.board[i][i][-1].player for i in range(4)):
                return self.board[0][0][-1].player

        # Check other diagonal
        if all(self.board[i][3 - i] for i in range(4)):
            if all(self.board[0][3][-1].player == self.board[i][3 - i][-1].player for i in range(4)):
                return self.board[0][3][-1].player

        return None

    # Keep these methods for terminal display if needed
    def get_board_display(self):
        """Return a string representation of the board."""
        result = "    0   1   2   3\n\n"

        # Draw the top border
        result += "  ┌"
        for col in range(3):
            result += "───┬"
        result += "───┐\n"

        for row in range(4):
            # Row number
            result += f"{row} │"

            # Cell contents
            for col in range(4):
                if not self.board[row][col]:
                    result += "   │"
                else:
                    result += f" {self.board[row][col][-1]} │"
            result += "\n"

            # Row separator, except after the last row
            if row < 3:
                result += "  ├"
                for col in range(3):
                    result += "───┼"
                result += "───┤\n"
            else:
                result += "  └"
                for col in range(3):
                    result += "───┴"
                result += "───┘\n"

        return result

    def display_stacks(self, player):
        """Return a string representation of a player's stacks."""
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
        """Select a move at random from legal moves."""
        legal_moves = game_state.get_legal_moves()
        if legal_moves:
            return random.choice(legal_moves)
        return None

class HueristicMDPBot:
    def __init__(self, player_id):
        self.player_id = player_id
        # TODO: initialize any PPO components here (models, optimizers, etc.)

    def get_rewards(self, game:GobbletGame):
        """
        Use reward function to select optimal move.
        """
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
        """Compute the reward from a ."""
        new_game = game.simulate_move(move)
        # First check if the move causes insta win or insta loss
        if new_game.check_winner() == self.player_id:
            return 1_000_000
        if new_game.check_winner() == 3 - self.player_id:
            return -1_000_000
        if self._opponent_can_win_in_one_move(new_game):
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
        return  score

    def _get_all_lines(self, game_state):
        """
        Return an array of "lines", each line is a list of up to 4 stacks from the board.
        Example: rows, columns, diagonals.
        """
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
        """
        If an opponent had a big threat in old_state, but in new_state we blocked it,
        award some points.
        """
        opponent = 3 - player
        old_threats = self._count_threats(old_game, opponent)
        new_threats = self._count_threats(new_game, opponent)
        if new_threats < old_threats:
            # We presumably blocked at least some threat
            return (old_threats - new_threats) * 0.5
        return 0.0

    def _count_threats(self, game_state, player):
        """
        Return a numeric 'threat level' for `player`.
        Example: +0.5 per "2 in a row" that is open, +2 per "3 in a row" that is open.
        "Open" means not blocked by bigger pieces of the opponent on that line.
        You can get more detailed if you want to check piece sizes thoroughly.
        """
        # Count possible lines (4 rows + 4 cols + 2 diagonals = 10 total)
        # We'll do a naive approach: for each line, see how many pieces of `player` are on top.
        # Then assign partial reward accordingly.
        threat_value = 0.0

        lines = self._get_all_lines(game_state)
        for line in lines:
            top_players = [cell[-1].player for cell in line if cell]  # who is on top
            count_me = top_players.count(player)
            if count_me == 2:
                threat_value += 0.5
            elif count_me == 3:
                threat_value += 2
            # You can detect 1 in a row or 4 in a row, etc.

        return threat_value

    def _opponent_can_win_in_one_move(self, game: GobbletGame) -> bool:
        valid_moves = game.get_legal_moves()
        players_turn = game.current_player
        for move in valid_moves:
            new_game = game.simulate_move(move)
            if players_turn == new_game.check_winner():
                return True
        return False

def simulate_games(bot1_class, bot2_class, num_games=10, verbose=True):
    """
    Simulate bot1 vs bot2 for `num_games` matches.
    If `verbose` is False, no board info is printed.
    """
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
                # No legal moves: force a break or treat as a pass.
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
    """Convert a cell to a vector representation."""
    vec = [0,0,0,0]
    for idx, piece in enumerate(cell):
        multiplier = 1 if piece.player == 1 else -1
        vec[idx] = multiplier * piece.size
    return vec

def stack_to_vector(stack):
    """Convert a stack to a vector representation."""
    vec = [0,0,0,0]
    for idx, piece in enumerate(stack):
        multiplier = 1 if piece.player == 1 else -1
        vec[idx] = multiplier * piece.size
    return vec

def vectorize_state(game):
    """
    Convert the board + stacks into a fixed-length vector.
    """
    state_vec = []

    # Player 1 stacks
    for stack in game.players[1]:
        stack_vector = stack_to_vector(stack)
        state_vec.append(stack_vector)

    # Player 2 stacks
    for stack in game.players[2]:
        stack_vector = stack_to_vector(stack)
        state_vec.append(stack_vector)

    # Board: 4x4
    for row in range(4):
        for col in range(4):
            cell_vector = cell_to_vector(game.board[row][col])
            state_vec.append(cell_vector)

    return state_vec

def vectorize_move(move):
    """
    move = (source_type, source_idx, target_x, target_y)
    We'll encode as:
    [source_type_is_stack(1 or 0),
     source_x, source_y,
     target_x, target_y].
    If source_type == 'stack', let source_x = stack_idx, source_y = -1 (arbitrary).
    """
    source_type, source_idx, tx, ty = move
    if source_type == "stack":
        return [1, source_idx, -1, tx, ty]
    else:
        sx, sy = source_idx
        return [0, sx, sy, tx, ty]

# TODO: FIX WINNING LOGIC FOR SAVING DATA WINNER VARIABLE IS FUCKED
def simulate_random_vs_random(num_games=10, save_data=False, output_file="game_data.csv"):
    """
    Simulate random vs. random and record (state, move, outcome).
    If save_data=True, write lines to `output_file`.
    """
    all_data = []  # Will hold tuples (state_vec, move_vec, outcome)

    for g in range(num_games):
        game = GobbletGame()
        bot1 = RandomBot(1)
        bot2 = RandomBot(2)
        outcome = None
        all_data_game = []

        while True:
            current_player = game.current_player
            state_vec = [[1 if current_player == 1 else -1]] + vectorize_state(game)

            # Bot move
            if current_player == 1:
                move = bot1.select_move(game.get_game_state())
            else:
                move = bot2.select_move(game.get_game_state())

            if move:
                game.make_move(move)
            else:
                # No moves possible, break
                break

            move_vec = vectorize_move(move)
            winner = game.check_winner()

            if winner:
                outcome = 1 if winner == 1 else -1
                break

            all_data_game.append((state_vec, move_vec))

        for state_information in all_data_game:
            all_data.append((state_information[0], state_information[1], outcome))


    # Save to CSV if requested
    if save_data:
        with open(output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            # Example CSV columns: state flattened + move vector + outcome
            for (s_vec, m_vec, res) in all_data:
                row = s_vec + m_vec + [res]
                writer.writerow(row)

    return all_data

def simulate_heuristic_vs_heuristic(num_games=10, save_data=False, output_file="game_data.csv"):
    """
    Simulate random vs. random and record (state, move, outcome).
    If save_data=True, write lines to `output_file`.
    """
    all_data = []  # Will hold tuples (state_vec, move_vec, outcome)

    for g in range(num_games):
        game = GobbletGame()
        bot1 = HueristicMDPBot(1)
        bot2 = HueristicMDPBot(2)
        outcome = None
        all_data_game = []

        while True:
            current_player = game.current_player
            state_vec = [[1 if current_player == 1 else -1]] + vectorize_state(game)

            # Bot move
            if current_player == 1:
                move = bot1.select_move(game.get_game_state())
            else:
                move = bot2.select_move(game.get_game_state())

            if move:
                game.make_move(move)
            else:
                # No moves possible, break
                break

            move_vec = vectorize_move(move)
            winner = game.check_winner()

            if winner:
                outcome = 1 if winner == 1 else -1
                break

            all_data_game.append((state_vec, move_vec))

        for state_information in all_data_game:
            all_data.append((state_information[0], state_information[1], outcome))


    # Save to CSV if requested
    if save_data:
        with open(output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            # Example CSV columns: state flattened + move vector + outcome
            for (s_vec, m_vec, res) in all_data:
                row = s_vec + m_vec + [res]
                writer.writerow(row)

    return all_data

def simulate_random_vs_hueristic(num_games=10, save_data=False, output_file="game_data.csv"):
    """
    Simulate random vs. random and record (state, move, outcome).
    If save_data=True, write lines to `output_file`.
    """
    all_data = []  # Will hold tuples (state_vec, move_vec, outcome)
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

            # Bot move
            if current_player == 1:
                move = bot1.select_move(game.get_game_state())
            else:
                move = bot2.select_move(game.get_game_state())

            if move:
                game.make_move(move)
            else:
                # No moves possible, break
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


    # Save to CSV if requested
    if save_data:
        with open(output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            # Example CSV columns: state flattened + move vector + outcome
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

