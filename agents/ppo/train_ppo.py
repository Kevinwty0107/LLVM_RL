import argparse
import os
from typing import Optional

from stable_baselines3 import PPO

from agents.env_maker import make_env


def parse_args():
    # Exp Related
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="name of experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of experiment")
    parser.add_argument("--bench-name", type=str, default='himeno',
                        help="benchmark to use")

    # Env Related
    parser.add_argument("--use-terminal-action", type=bool, default=True,
                        help="whether to use terminal action wrapper")
    parser.add_argument("--use-reward-difference", type=bool, default=True,
                        help="whether to use difference in reward")
    parser.add_argument("--reward-scaling", type=str, default='linear',
                        help="use cubic/exponential scaling to encourage risky behaviour")
    parser.add_argument("--denoise-threshold", type=int, default=-1,
                        help="denoise small rewards for numerical stability")
    parser.add_argument("--episode-timesteps", type=int, default=200,
                        help="hard limit on timesteps per episode")
    parser.add_argument("--action-history", type=int, default=40,
                        help="number of history actions to record")
    parser.add_argument("--encoding", type=str, default='ir2vec',
                        help="encoding for bitcode programs")
    parser.add_argument("--record-rollouts", type=str, default="ppo_rollouts",
                        help="location to save rollouts")

    # Agent Related
    parser.add_argument("--total-timesteps", type=int, default=20481,
                        help="total timesteps of experiment")
    parser.add_argument("--learning-rate", type=float, default=0.00025,
                        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=10000,
                        help="the replay memory buffer size")
    parser.add_argument("--n-steps", type=int, default=1024,
                        help="number of steps per update")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
                        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
                        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.8,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=2000,
                        help="timestep to start learning")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    env = make_env(args)

    terminal = "terminal" if args.use_terminal_action else args.episode_timesteps
    log_name = f'{args.exp_name}_{args.bench_name}_{terminal}_{args.action_history}_{args.n_steps}'
    print(f'Experiment: {log_name}')

    model = PPO("MlpPolicy", env, verbose=1, n_steps=args.n_steps, tensorboard_log=f'ppo/{log_name}')
    model.learn(total_timesteps=24001)
    model.save(log_name)

    # del model  # remove to demonstrate saving and loading
    #
    # model = PPO.load("ppo_llvm")

    print('===========Evaluating============')
    obs = env.reset()
    step = 0
    dones = False
    while not dones:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        print(f'Step {step}: Reward = {rewards}')
        step += 1
