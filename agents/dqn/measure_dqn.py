import pickle

from matplotlib import pyplot as plt
from stable_baselines3 import DQN

from agents.env_maker import make_env
from my_dqn_sb3 import parse_args

args = parse_args()
args.action_history = 40

model = DQN.load(f"trained/my_dqn_sb3_200_40_200")

for seed in [5, 6, 7, 8]:
    print(f'=========Evaluating: {seed}===========')
    args.seed = seed
    env = make_env(args)
    obs = env.reset()

    step = 0
    dones = False
    cum_reward = 0
    scores = [0]
    actions = [0]

    while not dones:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        print(f'Step {step}: Reward = {rewards}')
        cum_reward += rewards
        scores.append(cum_reward)
        actions.append(action)
        step += 1
    print(f'Cumulative Reward = {cum_reward}')
    plt.plot(scores, label=f'DQN, seed={seed}')

    with open(f'logs/eval/dqn_scorelog_{seed}', 'wb+') as f:
        pickle.dump((actions, scores), f)

plt.xlabel('Iterations')
plt.ylabel('Score (O0-O3 normalized)')
plt.legend()
plt.savefig(f'DQN_1.png')
