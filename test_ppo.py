from ppo import PPOBot
from ppo import simulate_and_train
from gobblet_game import GobbletGame, RandomBot, HueristicMDPBot

def train_and_evaluate():
    ppo_bot = PPOBot(
        player_id=1,
        state_dim=89,
        action_dim=288,
        hidden_dim=512,
        lr=3e-2,
        gamma=0.999,
        kl=0.002,
        gae=0.95,
        clip=0.1,
        norm=0.5,
        vf=1.0,
        ent=0.001
    )
    
    print("\nRand bot train")
    simulate_and_train(
        ppo_bot=ppo_bot,
        opponent_bot=RandomBot(player_id=2),
        num_games=1000,
        save_path='ppo_vs_random_initial.pt'
    )
    quit()
    print("\nRand bot train 2")
    ppo_bot.load_model('ppo_vs_random_initial.pt')
    simulate_and_train(
        ppo_bot=ppo_bot,
        opponent_bot=RandomBot(player_id=2),
        num_games=10000,
        save_path='ppo_vs_random_b.pt'
    )
    print("\nRand bot train 3")
    ppo_bot.load_model('ppo_vs_random_b.pt')
    simulate_and_train(
        ppo_bot=ppo_bot,
        opponent_bot=RandomBot(player_id=2),
        num_games=10000,
        save_path='ppo_vs_random_c.pt'
    )
    print("\nRand bot train 4")
    ppo_bot.load_model('ppo_vs_random_c.pt')
    simulate_and_train(
        ppo_bot=ppo_bot,
        opponent_bot=RandomBot(player_id=2),
        num_games=10000,
        save_path='ppo_vs_random_d.pt'
    )
    print("\nRand bot train 5")
    ppo_bot.load_model('ppo_vs_random_d.pt')
    simulate_and_train(
        ppo_bot=ppo_bot,
        opponent_bot=RandomBot(player_id=2),
        num_games=10000,
        save_path='ppo_vs_random_final.pt'
    )

if __name__ == "__main__":
    train_and_evaluate()


