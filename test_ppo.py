from ppo import PPOBot
from ppo import simulate_and_train
from gobblet_game import GobbletGame, RandomBot, HueristicMDPBot

def train_against_opp(pretrain=True):
    # Create PPO bot with improved architecture
    ppo_bot = PPOBot(
        player_id=1,
        state_dim=89,
        action_dim=288,  # Full action space (240 board moves + 48 stack moves)
        hidden_dim=256,  # Reduced network size for better stability
        lr=3e-4,  # Slightly higher learning rate for pre-training
        gamma=0.99,
        epsilon=0.2,
        c1=0.5,
        c2=0.01
    )
    
    # First pre-train on historical data if requested
    if pretrain:
        print("Pre-training on historical data...")
        ppo_bot.train_from_data('game_data2.csv', num_epochs=4, batch_size=16)
        ppo_bot.save_model('pretrained_model.pt')
        print("Pre-training complete!")
    
    # Test PPO bot against itself for 1000 online games
    print("\nTesting PPO bot against itself...")
    simulate_and_train(
        ppo_bot=ppo_bot,
        opponent_bot=ppo_bot,
        num_games=1000,
        save_path='final_model.pt',
        learn = True
    )

    # get results of PPO bot against RandomBot for 500 games
    print("\nTesting PPO bot against RandomBot...")
    simulate_and_train(
        ppo_bot=ppo_bot,
        opponent_bot=RandomBot(player_id=3),
        num_games=250,
        save_path='',
        learn = True
    )

if __name__ == "__main__":
    train_against_opp(pretrain=True)


