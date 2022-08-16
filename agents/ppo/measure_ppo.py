import pickle

from matplotlib import pyplot as plt
from stable_baselines3 import PPO

from agents.env_maker import make_env
from train_ppo import parse_args

args = parse_args()
args.action_history = 40

for n_steps in [2048]:

    model = PPO.load(f"trained/my_ppo_terminal_0_{n_steps}")

    for seed in [1, 2, 3, 4]:
        print(f'=========Evaluating: {n_steps}-{seed}==========')
        args.seed = seed
        env = make_env(args)
        model.set_random_seed(seed)
        obs = env.reset()

        step = 0
        dones = False
        cum_reward = 0
        scores = [0]
        actions = [0]

        while not dones:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            print(info)
            print(f'Step {step}: Reward = {rewards}')
            cum_reward += rewards
            scores.append(cum_reward)
            actions.append(action)
            step += 1
        print(f'Cumulative Reward = {cum_reward}')
        plt.plot(scores, label=f'PPO, seed={seed}')

        with open(f'logs/eval/ppo_scorelog_{n_steps}_{seed}_new', 'wb+') as f:
            pickle.dump((actions, scores), f)

    plt.xlabel('Iterations')
    plt.ylabel('Score (O0-O3 normalized)')
    plt.legend()
    plt.savefig(f'PPO_{n_steps}_new.png')
