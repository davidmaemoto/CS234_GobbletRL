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
from gobblet_game import GobbletGame, RandomBot, HueristicMDPBot

class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim=288, hidden_dim=512):
        super(PPONetwork, self).__init__()
        
        self.input_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.board_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()

    @staticmethod
    def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                self.init_layer(module)
        
        self.init_layer(self.policy_net[-1], std=0.01)
        self.init_layer(self.value_net[-1], std=1.0)
        
    def forward(self, x):
        x = self.input_net(x)
        board_features = self.board_net(x)
        policy_logits = self.policy_net(board_features)
        value = self.value_net(board_features)
        return policy_logits, value

class PPOBot:
    def __init__(self, player_id, state_dim=89, action_dim=288, hidden_dim=512, lr=3e-4, gamma=0.995, kl=0.02,
                 gae=0.97, clip=0.3, norm=1.0, vf=1.0, ent=0.01):
        self.player_id = player_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = PPONetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_old = PPONetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        
        self.gamma = gamma
        self.kl = kl
        self.gae = gae
        self.clip = clip
        self.norm = norm
        self.vf = vf
        self.ent = ent
        self.beta = 0.5

        self.trajs = []

        self.action_map = self._idx_to_action()
        
    def _idx_to_action(self):
        action_map = {}
        idx = 0
        
        for sx in range(4):
            for sy in range(4):
                for tx in range(4):
                    for ty in range(4):
                        if (sx, sy) != (tx, ty):
                            action_map[idx] = ("board", (sx, sy), tx, ty)
                            idx += 1
        for stack in range(3):
            for tx in range(4):
                for ty in range(4):
                    action_map[idx] = ("stack", stack, tx, ty)
                    idx += 1
                    
        assert len(action_map) == 288, f"Wrong number of actions"
        assert all(i in action_map for i in range(288)), "Wrong number of actions 2"
        
        return action_map
        
    def _move_to_index(self, move):
        source_type, source_idx, tx, ty = move
        
        if source_type == "board":
            sx, sy = source_idx
            if sx == tx and sy == ty:
                return None
            base = sx * 60 + sy * 15
            off = tx * 4 + ty
            if off >= sx * 4 + sy:
                off -= 1
            return base + off
        else:
            stack_num = source_idx
            return 240 + stack_num * 16 + tx * 4 + ty
            
    def _index_to_move(self, index):
        return self.action_map[index]

    def compute_gae(self, rewards, values, next_value):
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae * last_gae

        return (advantages - advantages.mean()) / (advantages.std() + 1e-8), advantages + values

    def train_on_episode(self, trajectories):
        if not trajectories:
            return

        states = torch.FloatTensor([t['state'] for t in trajectories]).to(self.device)
        acts_idxs = torch.LongTensor([t['action_idx'] for t in trajectories]).to(self.device)
        old_ps = torch.FloatTensor([t['prob'] for t in trajectories]).to(self.device)
        rs = torch.FloatTensor([t['reward'] for t in trajectories]).to(self.device)
        
        with torch.no_grad():
            _, values = self.policy_old(states)
            values = values.squeeze()
            if len(trajectories) > 1:
                next_state = torch.FloatTensor([trajectories[-1]['state']]).to(self.device)
                _, next_value = self.policy_old(next_state)
                next_value = next_value.squeeze()
            else:
                next_value = values[-1]

        advantages, returns = self.compute_gae(rs, values, next_value)
        
        batch_size = len(trajectories)
        mini_batch_size = min(64, batch_size)
        num_epochs = 8
        best_loss = float('inf')
        no_improvement_count = 0
        
        for epoch in range(num_epochs):
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_ent = 0
            num_minibatches = 0
            
            indices = torch.randperm(batch_size)
            
            for start_idx in range(0, batch_size, mini_batch_size):
                batch_indices = indices[start_idx:start_idx + mini_batch_size]
                
                mb_states = states[batch_indices]
                mb_actions = acts_idxs[batch_indices]
                mb_old_probs = old_ps[batch_indices]
                mb_advantages = advantages[batch_indices]
                mb_returns = returns[batch_indices]
                
                policy_logits, current_values = self.policy(mb_states)
                current_values = current_values.squeeze()
                
                probs = F.softmax(policy_logits, dim=-1)
                current_probs = probs.gather(1, mb_actions.unsqueeze(1)).squeeze()
                
                ratio = torch.exp(torch.log(current_probs + 1e-10) - torch.log(mb_old_probs + 1e-10))
                ratio = torch.clamp(ratio, 0.0, 10.0)
                
                policy_loss = -torch.min(
                    ratio * mb_advantages,
                    torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * mb_advantages
                ).mean()
                
                value_pred_clipped = values[batch_indices] + torch.clamp(
                    current_values - values[batch_indices],
                    -self.clip, self.clip
                )
                value_loss = torch.max(
                    F.mse_loss(current_values, mb_returns),
                    F.mse_loss(value_pred_clipped, mb_returns)
                )
                
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
                
                with torch.no_grad():
                    old_logits, _ = self.policy_old(mb_states)
                    old_probs_full = F.softmax(old_logits, dim=-1)
                
                kl_div = F.kl_div(F.log_softmax(policy_logits, dim=-1), old_probs_full, reduction='batchmean')
                loss = (policy_loss + self.vf * value_loss - self.ent * (entropy + 1e-8).sqrt() + self.beta * kl_div)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.norm)
                self.optimizer.step()
                
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_ent += entropy.item()
                num_minibatches += 1
                
                if kl_div > 4 * self.kl:
                    break
            
            if num_minibatches > 0:
                avg_loss = (epoch_policy_loss + epoch_value_loss) / num_minibatches
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= 3:
                        break
            
            if kl_div >= 1.5 * self.kl:
                self.beta *= 1.5
            elif kl_div <= self.kl / 1.5:
                self.beta /= 1.5
            
            self.beta = torch.clamp(torch.tensor(self.beta), 0.1, 10.0).item()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.trajs.clear()

    def collect_trajectory(self, game_state, bot_type):
        legal_moves = game_state.get_legal_moves()
        if not legal_moves:
            return None

        state = self._vectorize_state(game_state)
        s_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, value = self.policy(s_tensor)
            
        legal_indices = [self._move_to_index(move) for move in legal_moves if self._move_to_index(move) is not None]
        
        action_mask = torch.zeros(288, dtype=torch.bool)
        action_mask[legal_indices] = True
        
        logits = policy_logits.squeeze()
        logits[~action_mask] = float('-inf')
        probs = F.softmax(logits, dim=0)
        
        try:
            move_idx = torch.multinomial(probs, 1).item()
            chosen_move = self._index_to_move(move_idx)
            prob = probs[move_idx].item()
        except RuntimeError:
            move_idx = random.choice(legal_indices)
            chosen_move = self._index_to_move(move_idx)
            prob = 1.0 / len(legal_indices)
        
        if bot_type == "ppo":
            self.trajs.append({
                'state': state,
                'action_idx': move_idx,
                'prob': prob,
                'value': value.item()
            })
        return chosen_move

    def _vectorize_state(self, game_state):
        state_vec = []
        state_vec.append([1 if game_state.current_player == 1 else -1])
        state_vec.extend(self._vectorize_board_and_stacks(game_state))
        return np.concatenate(state_vec)
        
    def _vectorize_board_and_stacks(self, game_state):
        state_vec = []
        for stack in game_state.players[1]:
            stack_vector = self._stack_to_vector(stack)
            state_vec.append(stack_vector)
        for stack in game_state.players[2]:
            stack_vector = self._stack_to_vector(stack)
            state_vec.append(stack_vector)
        for row in range(4):
            for col in range(4):
                cell_vector = self._cell_to_vector(game_state.board[row][col])
                state_vec.append(cell_vector)
        return state_vec
        
    def _stack_to_vector(self, stack):
        vec = [0, 0, 0, 0]
        for idx, piece in enumerate(stack):
            vec[idx] = piece.size if piece.player == 1 else -1 * piece.size
        return vec
        
    def _cell_to_vector(self, cell):
        vec = [0, 0, 0, 0]
        for idx, piece in enumerate(cell):
            vec[idx] = piece.size if piece.player == 1 else -1 * piece.size
        return vec
        
    def _vectorize_move(self, move):
        move_type, stack_num, x, y = move
        if move_type == "stack":
            return [move_type, -1, x, y]
        else:
            from_x, from_y = stack_num
            return [from_x, from_y, x, y]
            
    def _compute_returns(self, rewards):
        returns = torch.zeros_like(rewards)
        r2 = 0
        for t in reversed(range(len(rewards))):
            r2 = rewards[t] + self.gamma * r2
            returns[t] = r2
        return returns
        
    def compute_reward(self, old_state, new_state, action):
        reward = 0.0

        opponent_id = 3 - self.player_id
        can_opponent_win = self._can_win_in_one_move(new_state, opponent_id)
        if can_opponent_win:
            return -10000.0

        could_opponent_win = self._can_win_in_one_move(old_state, opponent_id)
        if could_opponent_win and not can_opponent_win:
            reward += 100.0
        
        winner = new_state.check_winner()
        if winner:
            return 10000000.0 if winner == self.player_id else -10000000.0
        
        center_positions = [(1,1), (1,2), (2,1), (2,2)]
        center_control = 0
        for x, y in center_positions:
            cell = new_state.board[x][y]
            if cell and cell[-1].player == self.player_id:
                center_control += 1
        reward += 0.5 * center_control
        
        lines = self._get_all_lines(new_state)
        threat_value = 0
        for line in lines:
            my_pieces = sum(1 for cell in line if cell and cell[-1].player == self.player_id)
            opp_pieces = sum(1 for cell in line if cell and cell[-1].player != self.player_id)
            if my_pieces == 2 and opp_pieces == 0:
                threat_value += 2.0
            elif my_pieces == 3:
                threat_value += 5.0
        reward += threat_value
        
        opp_threat_value = 0
        for line in lines:
            opp_pieces = sum(1 for cell in line if cell and cell[-1].player != self.player_id)
            my_pieces = sum(1 for cell in line if cell and cell[-1].player == self.player_id)
            if opp_pieces == 2 and my_pieces == 0:
                opp_threat_value += 2.0
            elif opp_pieces == 3:
                opp_threat_value += 5.0
        reward -= opp_threat_value

        move_type, from_pos, x, y = action
        if move_type == "stack":
            piece_size = len(old_state.players[self.player_id][from_pos])
            reward += 0.2 * piece_size

            target_cell = old_state.board[x][y]
            if target_cell and target_cell[-1].player != self.player_id:
                reward += 0.5 * piece_size

        return reward

    def _can_win_in_one_move(self, state, player_id):
        original_player = state.current_player
        state.current_player = player_id
        legal_moves = state.get_legal_moves()
        state.current_player = original_player

        for move in legal_moves:
            test_state = deepcopy(state)
            test_state.current_player = player_id
            if test_state.make_move(move):
                if test_state.check_winner() == player_id:
                    return True
        return False

    def _check_capture(self, old_state, new_state, action):
        _, _, x, y = action
        before_stack = old_state.board[x][y]
        after_stack = new_state.board[x][y]
        if len(before_stack) < len(after_stack):
            if before_stack and before_stack[-1].player != self.player_id:
                return 1.0
        return 0.0

    def _count_threats(self, game_state, player):
        threat_value = 0.0
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
        opponent = 3 - player
        old = self._count_threats(old_game, opponent)
        new = self._count_threats(new_game, opponent)
        diff = old - new
        if new < old:
            return diff * 0.5
        return 0.0
        
    def _get_all_lines(self, game_state):
        board = game_state.board
        lines = []
        for r in range(4):
            lines.append([board[r][c] for c in range(4)])
        for c in range(4):
            lines.append([board[r][c] for r in range(4)])
        lines.append([board[i][i] for i in range(4)])
        lines.append([board[i][3-i] for i in range(4)])
        return lines
        
    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)
        
    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.policy_old.load_state_dict(self.policy.state_dict())   

def simulate_and_train(ppo_bot, opponent_bot, num_games=100, save_path=None):
    wins = {1: 0, 2: 0}
    for game_idx in range(num_games):
        game = GobbletGame()
        trajectories = []
        moves_made = 0
        
        while True:
            current_player = game.current_player
            old_state = game.get_game_state()
            
            if current_player == 1:
                move = ppo_bot.collect_trajectory(old_state, "ppo")
            else:
                if isinstance(opponent_bot, RandomBot):
                    move = opponent_bot.select_move(old_state)
                elif isinstance(opponent_bot, HueristicMDPBot):
                    move = opponent_bot.select_move(old_state)
                else:
                    move = ppo_bot.collect_trajectory(old_state, "opponent")
            if not move:
                break
                
            success = game.make_move(move)
            moves_made += 1
            
            if success and current_player == 1:
                reward = ppo_bot.compute_reward(old_state, game, move)
                
                if len(ppo_bot.trajs) > 0:
                    ppo_bot.trajs[-1]['reward'] = reward
                    trajectories.append(ppo_bot.trajs[-1])
            
            winner = game.check_winner()
            if winner:
                wins[winner] += 1
                final_reward = 10.0 if winner == 1 else -10.0
                for t in trajectories:
                        t['reward'] = final_reward
                break
            
            if moves_made > 200:
                for t in trajectories:
                        t['reward'] = -1.0
                break
            
        if trajectories:
            ppo_bot.train_on_episode(trajectories)
        
        if (game_idx + 1) % 10 == 0:
            win_rate = wins[1] / (game_idx + 1) * 100
            print(f"G {game_idx + 1}/{num_games} - W%: {win_rate:.1f}% (Ppo: {wins[1]}, Opp: {wins[2]})")
    
    if save_path:
        ppo_bot.save_model(save_path)
    
    final_win_rate = wins[1] / num_games * 100
    print(f"\n{final_win_rate:.1f}%")
    print(f"Ws - PPO: {wins[1]}, Opp: {wins[2]}")
