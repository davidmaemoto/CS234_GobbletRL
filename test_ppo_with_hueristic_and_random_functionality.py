from ppo import PPOBot
from ppo import simulate_and_train
from ppo import simulate_games
from gobblet_game import GobbletGame, RandomBot, HueristicMDPBot
import matplotlib.pyplot as plt

def train_and_evaluate(random_not_hueristic=True):
    ppo_bot = PPOBot(
        player_id=1,
        state_dim=89,
        action_dim=288,
        hidden_dim=512,
        lr=3e-3 if random_not_hueristic else .001,
        gamma=0.99 if random_not_hueristic else .999,
        kl=0.02,
        gae=0.95,
        clip=0.1,
        norm=1,
        vf=1.0,
        ent=0.001
    )
    win_rates = []
    games_played = []
    for i in range(20):
        if i != 0:
            ppo_bot.load_model(f'ppo_vs_random_{i-1}.pt' if random_not_hueristic else f'ppo_vs_hueristic_{i-1}.pt')
        training_bot = "RandomBot" if random_not_hueristic else "HueristicMDPBot"
        print(f"Train {training_bot} {i}")
        win_rate = simulate_and_train(
            ppo_bot=ppo_bot,
            opponent_bot=RandomBot(player_id=2) if random_not_hueristic else HueristicMDPBot(player_id=2),
            num_games=250,
            save_path=f'ppo_vs_random_{i}.pt' if random_not_hueristic else f'ppo_vs_hueristic_{i}.pt'
        )
        win_rates.append(win_rate)
        games_played.append(500 * (i + 1))

    plt.title("PPO Bot Win Rate During Training")
    plt.xlabel("Games Played")
    plt.ylabel("Win Rate")
    plt.plot(games_played, win_rates)
    plt.show()

def manual_test_ppo(enhance=False):
    ppo_bot = PPOBot(
        player_id=1,
        state_dim=89,
        action_dim=288,
        hidden_dim=512,
        lr=3e-4,
        gamma=0.99,
        kl=0.02,
        gae=0.95,
        clip=0.1,
        norm=0.5,
        vf=1.0,
        ent=0.001
    )
    if enhance:
        ppo_bot.load_model('ppo_vs_hueristic_final.pt')
    else:
        ppo_bot.load_model('ppo_vs_random_final.pt')
    # simulate_games(ppo_bot, RandomBot(2), enhance_with_greedy=enhance)
    simulate_games(ppo_bot, HueristicMDPBot(2), num_games=100)
    simulate_games(ppo_bot, HueristicMDPBot(2), num_games=100, enhance_with_greedy=True)
    simulate_games(ppo_bot, RandomBot(2), num_games=100)


if __name__ == "__main__":
    train_and_evaluate(True)
    # manual_test_ppo(enhance=True, enhance_with_greedy=True)

