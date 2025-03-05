import random
from copy import deepcopy
from flask import Flask, render_template, request, jsonify, send_from_directory

# Import your existing game classes
from gobblet_game import GobbletGame, GamePiece, RandomBot, HueristicMDPBot
from ppo import PPOBot

app = Flask(__name__)

# Store game state in memory (for simplicity)
# In a production app, you'd want to use sessions or a database
active_game = None


@app.route('/')
def index():
    """Serve the main game page"""
    return render_template('index.html')


@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)


@app.route('/api/new-game', methods=['POST'])
@app.route('/api/new-game', methods=['POST'])
def new_game():
    """Create a new game, optionally against a bot"""
    global active_game
    data = request.json  # Check if a vsBot flag was sent
    vs_bot = data.get('vsBot', False)  # Default to False if not provided
    active_game = GobbletGame()
    active_game.vs_bot = vs_bot  # Store bot mode in the game instance
    return get_game_state()


@app.route('/api/game-state', methods=['GET'])
def get_game_state():
    """Get the current game state"""
    global active_game
    if not active_game:
        active_game = GobbletGame()

    # Convert game state to JSON-serializable format
    game_state = {
        'board': [[[piece_to_dict(piece) for piece in cell] for cell in row] for row in active_game.board],
        'players': {
            '1': [[piece_to_dict(piece) for piece in stack] for stack in active_game.players[1]],
            '2': [[piece_to_dict(piece) for piece in stack] for stack in active_game.players[2]]
        },
        'currentPlayer': active_game.current_player,
        'winner': active_game.check_winner()
    }

    return jsonify(game_state)


@app.route('/api/make-move', methods=['POST'])
def make_move():
    """Make a move in the game"""
    global active_game
    if not active_game:
        return jsonify({'error': 'No active game'}), 400

    data = request.json
    source_type = data.get('sourceType')
    source_idx = data.get('sourceIdx')
    target_x = data.get('targetX')
    target_y = data.get('targetY')

    if not all([source_type, source_idx is not None, target_x is not None, target_y is not None]):
        return jsonify({'error': 'Invalid move data'}), 400

    move = (source_type, source_idx, target_x, target_y) if source_type == "stack" else (source_type, tuple(source_idx), target_x, target_y)

    success = active_game.make_move(move)

    if success:
        winner = active_game.check_winner()
        if not winner and active_game.vs_bot and active_game.current_player == 2:
            # bot = RandomBot(2)
            #bot = HueristicMDPBot(2)
            bot = PPOBot(2)
            bot.load_model('final_model.pt')
            bot_move = bot.select_move(active_game.get_game_state())
            if bot_move:
                active_game.make_move(bot_move)

    return get_game_state()


@app.route('/api/valid-moves', methods=['POST'])
def get_valid_moves():
    """Get valid moves for a selected piece"""
    global active_game
    if not active_game:
        return jsonify({'error': 'No active game'}), 400

    data = request.json
    source_type = data.get('sourceType')
    source_idx = data.get('sourceIdx')

    # Validate input
    if not all([source_type, source_idx is not None]):
        return jsonify({'error': 'Invalid source data'}), 400

    # Get valid moves based on source type
    valid_moves = []
    if source_type == "stack":
        # Find all stack moves for this specific stack
        for move in active_game.get_stack_moves():
            if move[1] == source_idx:  # If the stack index matches
                valid_moves.append((move[2], move[3]))  # Add target coordinates
    else:  # board
        source_x, source_y = source_idx
        # Find all board moves for this specific position
        for move in active_game.get_board_moves():
            if move[1] == (source_x, source_y):  # If the source position matches
                valid_moves.append((move[2], move[3]))  # Add target coordinates

    return jsonify({'validMoves': valid_moves})


@app.route('/api/bot-move', methods=['POST'])
def make_bot_move():
    """Make a move as the bot"""
    global active_game
    if not active_game:
        return jsonify({'error': 'No active game'}), 400

    # Create a bot for the current player
    # bot = RandomBot(active_game.current_player)
    #bot = HueristicMDPBot(active_game.current_player)
    bot = PPOBot(active_game.current_player)
    bot.load_model('final_model.pt')
    # Get and make bot move
    bot_move = bot.select_move(active_game.get_game_state())
    if bot_move:
        active_game.make_move(bot_move)
        return get_game_state()
    else:
        return jsonify({'error': 'No valid bot moves'}), 400


def piece_to_dict(piece):
    """Convert a GamePiece object to a dictionary"""
    return {
        'size': piece.size,
        'player': piece.player
    }


if __name__ == '__main__':
    app.run(debug=True)