import math
import numpy as np
import random
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import csv
from gobblet_game import GobbletGame

class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim=288, hidden_dim=256):
        super(PPONetwork, self).__init__()
        # Input layer with layer normalization
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        # Hidden layers with layer normalization
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        # Separate heads for policy and value
        # Policy head outputs logits for each possible action
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Value head for state value estimation
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using orthogonal initialization"""
        for module in [self.fc1, self.fc2, self.fc3, self.actor, self.critic]:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)
        
    def forward(self, x):
        # Forward pass through shared layers with layer normalization and residual connections
        h1 = F.relu(self.ln1(self.fc1(x)))
        h2 = F.relu(self.ln2(self.fc2(h1))) + h1
        h3 = F.relu(self.ln3(self.fc3(h2))) + h2
        
        # Policy head
        policy_logits = self.actor(h3)
        
        # Value head
        value = self.critic(h3)
        
        return policy_logits, value

class PPOBot:
    def __init__(self, player_id, state_dim=89, action_dim=288, hidden_dim=256, lr=3e-5, gamma=0.99, epsilon=0.2, c1=0.5, c2=0.01):
        """
        :param player_id: Bot's player number (1 or 2)
        :param state_dim: Dimension of flattened state vector (89 for Gobblet)
        :param action_dim: Dimension of action vector (288 for full action space)
        :param hidden_dim: Hidden layer size (increased for more capacity)
        :param lr: Learning rate (reduced for stability)
        :param gamma: Discount factor
        :param epsilon: PPO clipping parameter
        :param c1: Value loss coefficient
        :param c2: Entropy coefficient for exploration
        """
        self.player_id = player_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks with larger capacity
        self.policy = PPONetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_old = PPONetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Use AdamW optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=lr,
            eps=1e-5,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler for adaptive learning
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # PPO hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        
        # Memory with increased capacity
        self.memory = deque(maxlen=2000)
        
        # Action space mapping
        self.action_map = self._create_action_map()
        
    def _create_action_map(self):
        """Create a mapping between action indices and game moves"""
        action_map = {}
        idx = 0
        
        # Board to board moves (16x15=240 possible moves)
        for sx in range(4):
            for sy in range(4):
                for tx in range(4):
                    for ty in range(4):
                        if (sx, sy) != (tx, ty):  # Exclude same position moves
                            action_map[idx] = ("board", (sx, sy), tx, ty)
                            idx += 1
                        
        # Stack to board moves (3x16=48 possible moves)
        for stack in range(3):  # 3 stacks
            for tx in range(4):  # 4x4 board
                for ty in range(4):
                    action_map[idx] = ("stack", stack, tx, ty)
                    idx += 1
                    
        # Verify we have exactly 288 moves
        assert len(action_map) == 288, f"Expected 288 moves, got {len(action_map)}"
        # Verify we have all indices from 0 to 287
        assert all(i in action_map for i in range(288)), "Missing some action indices"
        
        return action_map
        
    def _move_to_index(self, move):
        """Convert a move to its corresponding index in the action space"""
        source_type, source_idx, tx, ty = move
        
        if source_type == "board":
            sx, sy = source_idx
            if sx == tx and sy == ty:
                return None  # Invalid move
            # Calculate index for board moves
            # First calculate base index for the source position
            base = sx * 60 + sy * 15  # Each source position can move to 15 other positions
            # Then add offset for target position
            target_offset = tx * 4 + ty
            # Adjust for the skipped same-position move
            if target_offset >= sx * 4 + sy:
                target_offset -= 1
            return base + target_offset
        else:  # stack move
            stack_num = source_idx
            # Stack moves start after all board moves (240 board moves)
            return 240 + stack_num * 16 + tx * 4 + ty
            
    def _index_to_move(self, index):
        """Convert an action index to a game move"""
        if index not in self.action_map:
            raise ValueError(f"Invalid action index: {index}")
        return self.action_map[index]

    def select_move(self, game_state):
        """Select a move using the current policy"""
        legal_moves = game_state.get_legal_moves()
        if not legal_moves:
            return None

        # Vectorize state
        state = self._vectorize_state(game_state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get policy logits and value
        with torch.no_grad():
            policy_logits, value = self.policy_old(state_tensor)
            
        # Convert legal moves to action indices
        legal_indices = []
        for move in legal_moves:
            idx = self._move_to_index(move)
            if idx is not None:
                legal_indices.append(idx)
        
        # Create mask for legal moves
        action_mask = torch.zeros(288, dtype=torch.bool)
        action_mask[legal_indices] = True
        
        # Apply mask and get probabilities
        logits = policy_logits.squeeze()
        logits[~action_mask] = float('-inf')  # Mask illegal moves
        probs = F.softmax(logits, dim=0)
        
        # Sample move using probabilities
        try:
            move_idx = torch.multinomial(probs, 1).item()
            chosen_move = self._index_to_move(move_idx)
            prob = probs[move_idx].item()
        except RuntimeError:
            # Fallback to random selection if sampling fails
            move_idx = random.choice(legal_indices)
            chosen_move = self._index_to_move(move_idx)
            prob = 1.0 / len(legal_indices)
        
        # Store transition with action index instead of move vector
        self.memory.append({
            'state': state,
            'action_idx': move_idx,
            'prob': prob,
            'value': value.item()
        })
        
        return chosen_move

    def train_on_episode(self, transitions):
        """Update policy using PPO"""
        if not transitions:
            return
            
        # Convert transitions to tensors
        states = torch.FloatTensor([t['state'] for t in transitions]).to(self.device)
        action_indices = torch.LongTensor([t['action_idx'] for t in transitions]).to(self.device)
        old_probs = torch.FloatTensor([t['prob'] for t in transitions]).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in transitions]).to(self.device)
        
        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Compute value targets and advantages
        with torch.no_grad():
            _, values = self.policy(states)
            values = values.squeeze()
            advantages = rewards - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(3):  # Reduced number of epochs for stability
            # Get current policy and value predictions
            policy_logits, current_values = self.policy(states)
            
            # Calculate action probabilities
            probs = F.softmax(policy_logits, dim=-1)
            current_probs = probs.gather(1, action_indices.unsqueeze(1)).squeeze()
            
            # Calculate probability ratio
            ratio = current_probs / (old_probs + 1e-10)
            ratio = torch.clamp(ratio, 0.0, 10.0)
            
            # Calculate surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss using Huber loss for robustness
            value_loss = F.smooth_l1_loss(current_values.squeeze(), rewards)
            
            # Entropy loss for exploration
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            
            # Total loss
            loss = policy_loss + self.c1 * value_loss - self.c2 * entropy
            
            # Skip update if loss is invalid
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            # Update learning rate based on loss
            self.scheduler.step(loss)
        
        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear()

    def _vectorize_state(self, game_state):
        """Convert game state to vector format"""
        state_vec = []
        
        # Current player indicator
        state_vec.append([1 if game_state.current_player == 1 else -1])
        
        # Add board and stack vectors
        state_vec.extend(self._vectorize_board_and_stacks(game_state))
        
        return np.concatenate(state_vec)
        
    def _vectorize_board_and_stacks(self, game_state):
        """Helper method to vectorize board and stacks"""
        state_vec = []
        
        # Player 1 stacks
        for stack in game_state.players[1]:
            stack_vector = self._stack_to_vector(stack)
            state_vec.append(stack_vector)
            
        # Player 2 stacks
        for stack in game_state.players[2]:
            stack_vector = self._stack_to_vector(stack)
            state_vec.append(stack_vector)
            
        # Board
        for row in range(4):
            for col in range(4):
                cell_vector = self._cell_to_vector(game_state.board[row][col])
                state_vec.append(cell_vector)
                
        return state_vec
        
    def _stack_to_vector(self, stack):
        """Convert a stack to vector representation"""
        vec = [0, 0, 0, 0]
        for idx, piece in enumerate(stack):
            multiplier = 1 if piece.player == 1 else -1
            vec[idx] = multiplier * piece.size
        return vec
        
    def _cell_to_vector(self, cell):
        """Convert a cell to vector representation"""
        vec = [0, 0, 0, 0]
        for idx, piece in enumerate(cell):
            multiplier = 1 if piece.player == 1 else -1
            vec[idx] = multiplier * piece.size
        return vec
        
    def _vectorize_move(self, move):
        """Convert a move to vector format"""
        source_type, source_idx, tx, ty = move
        if source_type == "stack":
            return [source_idx, -1, tx, ty]  # 4-dimensional vector
        else:
            sx, sy = source_idx
            return [sx, sy, tx, ty]  # 4-dimensional vector
            
    def _compute_returns(self, rewards):
        """Compute discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
            
        return returns
        
    def compute_reward(self, old_game_state, new_game_state, action):
        """Compute reward for a state transition"""
        reward = 0.0

        # Win/loss rewards
        winner = new_game_state.check_winner()
        if winner:
            return 10.0 if winner == self.player_id else -10.0
            
        # Capture reward
        reward += self._check_capture(old_game_state, new_game_state, action)

        # Threat-based rewards
        reward += self._count_threats(new_game_state, self.player_id)
        reward += self._count_blocked_threats(old_game_state, new_game_state, self.player_id)
        reward -= self._count_threats(new_game_state, 3 - self.player_id) * 0.5

        return reward

    def _check_capture(self, old_state, new_state, action):
        """Check if a move resulted in capturing an opponent's piece"""
        _, _, tx, ty = action
        before_stack = old_state.board[tx][ty]
        after_stack = new_state.board[tx][ty]

        if len(before_stack) < len(after_stack):
            if before_stack and before_stack[-1].player != self.player_id:
                return 1.0
        return 0.0

    def _count_threats(self, game_state, player):
        """Count the number of threatening positions"""
        threat_value = 0.0
        
        # Get all lines (rows, columns, diagonals)
        lines = self._get_all_lines(game_state)
        
        for line in lines:
            top_players = [cell[-1].player for cell in line if cell]
            count_me = top_players.count(player)
            if count_me == 2:
                threat_value += 0.5
            elif count_me == 3:
                threat_value += 2.0
                
        return threat_value
        
    def _count_blocked_threats(self, old_game, new_game, player):
        """Compute reward for blocking opponent threats"""
        opponent = 3 - player
        old_threats = self._count_threats(old_game, opponent)
        new_threats = self._count_threats(new_game, opponent)
        
        if new_threats < old_threats:
            return (old_threats - new_threats) * 0.5
        return 0.0
        
    def _get_all_lines(self, game_state):
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
        
    def save_model(self, path):
        """Save the policy network"""
        torch.save(self.policy.state_dict(), path)
        
    def load_model(self, path):
        """Load a saved policy network"""
        self.policy.load_state_dict(torch.load(path))
        self.policy_old.load_state_dict(self.policy.state_dict())   

        
    def train_from_data(self, data_path, num_epochs=5, batch_size=32):
        """Train PPO bot using historical game data"""
        print("Starting training from historical data...")
        
        # Load and preprocess data
        transitions = load_training_data(data_path)
        if not transitions:
            return
            
        # Convert moves to action indices
        processed_transitions = []
        for t in transitions:
            action = t['action']
            if action['source_type'] == 'board':
                tx, ty = action['target']
                move = ('board', action['source_idx'], tx, ty)
            else:
                tx, ty = action['target']
                move = ('stack', action['source_idx'], tx, ty)
            idx = self._move_to_index(move)
            if idx is not None:
                t['action_idx'] = idx
                processed_transitions.append(t)
        
        # Convert to tensors
        states = torch.FloatTensor([t['state'] for t in processed_transitions]).to(self.device)
        action_indices = torch.LongTensor([t['action_idx'] for t in processed_transitions]).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in processed_transitions]).to(self.device)
        
        # Remove any NaN values
        valid_mask = ~torch.isnan(rewards) & ~torch.isnan(states).any(dim=1)
        states = states[valid_mask]
        action_indices = action_indices[valid_mask]
        rewards = rewards[valid_mask]
        
        # Normalize rewards
        rewards = torch.clamp(rewards, -10.0, 10.0)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        dataset_size = len(states)
        indices = list(range(dataset_size))
        best_loss = float('inf')
        patience = 5
        no_improvement = 0
        
        for epoch in range(num_epochs):
            random.shuffle(indices)
            total_loss = 0
            num_batches = 0
            
            for start_idx in range(0, dataset_size, batch_size):
                batch_indices = indices[start_idx:min(start_idx + batch_size, dataset_size)]
                
                batch_states = states[batch_indices]
                batch_actions = action_indices[batch_indices]
                batch_rewards = rewards[batch_indices]
                
                try:
                    # Get policy predictions
                    policy_logits, values = self.policy(batch_states)
                    
                    # Calculate action probabilities
                    probs = F.softmax(policy_logits, dim=-1)
                    chosen_probs = probs.gather(1, batch_actions.unsqueeze(1)).squeeze()
                    
                    # Get old policy predictions
                    with torch.no_grad():
                        old_logits, _ = self.policy_old(batch_states)
                        old_probs = F.softmax(old_logits, dim=-1)
                        old_chosen_probs = old_probs.gather(1, batch_actions.unsqueeze(1)).squeeze()
                    
                    # Calculate advantages
                    advantages = batch_rewards - values.squeeze()
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    # Calculate probability ratio
                    ratio = chosen_probs / (old_chosen_probs + 1e-10)
                    ratio = torch.clamp(ratio, 0.0, 10.0)
                    
                    # Calculate losses
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    value_loss = F.smooth_l1_loss(values.squeeze(), batch_rewards)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
                    
                    loss = policy_loss + self.c1 * value_loss - self.c2 * entropy
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    
                    # Optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except RuntimeError as e:
                    print(f"Warning: {str(e)}")
                    continue
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
                
                # Early stopping check
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    no_improvement = 0
                else:
                    no_improvement += 1
                    if no_improvement >= patience:
                        print("Early stopping triggered!")
                        break
            
            # Update old policy
            self.policy_old.load_state_dict(self.policy.state_dict())
            
            # Update learning rate
            self.scheduler.step(avg_loss)
        
        print("Training complete!")

def load_training_data(file_path):
    """Load training data from CSV file containing (s,a,r,s') tuples"""
    print(f"Loading training data from {file_path}...")
    
    data = []
    with open(file_path, mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Convert strings to floats
            row = [float(x) for x in row]
            
            # Extract components
            state = row[:89]  # First 89 elements are the state vector
            move_vec = row[89:94]  # 5 elements: [source_type, source_x, source_y, target_x, target_y]
            reward = row[94]  # Reward
            next_state = row[95:184]  # Next state vector
            
            # Convert move vector to action format
            source_type = 'board' if move_vec[0] == 0 else 'stack'
            if source_type == 'board':
                source_idx = (int(move_vec[1]), int(move_vec[2]))
            else:
                source_idx = int(move_vec[1])
            target = (int(move_vec[3]), int(move_vec[4]))
            
            data.append({
                'state': state,
                'action': {
                    'source_type': source_type,
                    'source_idx': source_idx,
                    'target': target
                },
                'reward': reward,
                'next_state': next_state
            })
    
    print(f"Loaded {len(data)} transitions")
    return data

def simulate_and_train(ppo_bot, opponent_bot, num_games=100, save_path=None, learn = True):
    """Train PPO bot against any opponent bot"""
    print(f"Starting self-play training against {opponent_bot.__class__.__name__}...")
    
    # Track wins for statistics
    wins = {1: 0, 2: 0}
    
    for game_idx in range(num_games):
        game = GobbletGame()
        game_memory = []
        moves_made = 0
        
        while True:
            current_player = game.current_player
            current_bot = ppo_bot if current_player == 1 else opponent_bot
            
            # Get current state before move
            old_state = game.get_game_state()
            
            # Get move from appropriate bot
            move = current_bot.select_move(old_state)
            if not move:
                break
                
            # Make the move
            success = game.make_move(move)
            moves_made += 1
            
            if success and current_player == 1:  # Only store PPO bot's moves
                # Calculate immediate reward
                reward = ppo_bot.compute_reward(old_state, game, move)
                
                # Store transition if it's in memory
                if len(ppo_bot.memory) > 0:
                    game_memory.append({
                        'transitions': list(ppo_bot.memory),
                        'reward': reward
                    })
            
            # Check for game end
            winner = game.check_winner()
            if winner:
                wins[winner] += 1
                # Update final rewards based on game outcome
                final_reward = 10.0 if winner == 1 else -10.0
                for mem in game_memory:
                    for t in mem['transitions']:
                        t['reward'] = final_reward
                break
            
            # If game is too long, end it with a small negative reward
            if moves_made > 200:
                for mem in game_memory:
                    for t in mem['transitions']:
                        t['reward'] = -1.0
                break
            
        if learn:
            # Train PPO bot on collected experience
            for mem in game_memory:
                ppo_bot.train_on_episode(mem['transitions'])
        
        # Print progress and statistics
        if (game_idx + 1) % 10 == 0:
            win_rate = wins[1] / (game_idx + 1) * 100
            print(f"Game {game_idx + 1}/{num_games} - Win Rate: {win_rate:.1f}% (PPO: {wins[1]}, Opponent: {wins[2]})")
    
    # Save the trained model
    if save_path:
        print(f"Saving model to {save_path}")
        ppo_bot.save_model(save_path)
    
    final_win_rate = wins[1] / num_games * 100
    print(f"\nTraining complete! Final win rate: {final_win_rate:.1f}%")
    print(f"Total wins - PPO: {wins[1]}, Opponent: {wins[2]}")
