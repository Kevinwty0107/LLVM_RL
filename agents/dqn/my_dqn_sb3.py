import argparse
import os

from stable_baselines3 import DQN

from agents.env_maker import make_env


def parse_args():
    # Exp Related
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="name of experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of experiment")
    parser.add_argument("--bench-name", type=str, default='coremark',
                        help="benchmark to use")

    # Env Related
    parser.add_argument("--use-terminal-action", type=bool, default=False,
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
                        help='encoding for bitcode programs')

    # Agent Related
    parser.add_argument("--total-timesteps", type=int, default=20001,
                        help="total timesteps of experiment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=10000,
                        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--target-update-frequency", type=int, default=200,
                        help="timesteps to update the target network")
    # parser.add_argument("--max-grad-norm", type=float, default=0.5,
    #                     help="the maximum norm for the gradient clipping")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
                        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
                        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.8,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=2000,
                        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=1,
                        help="the frequency of training")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    env = make_env(args)

    terminal = "terminal" if args.use_terminal_action else args.episode_timesteps
    log_name = f'{args.exp_name}_{terminal}_{args.action_history}_{args.target_update_frequency}'
    print(f'Experiment: {log_name}')

    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_name, learning_rate=args.learning_rate,
                learning_starts=args.learning_starts, buffer_size=args.buffer_size, batch_size=args.batch_size,
                exploration_fraction=args.exploration_fraction, train_freq=args.train_frequency,
                target_update_interval=args.target_update_frequency)
    model.learn(total_timesteps=args.total_timesteps, log_interval=1)
    model.save(log_name)

    # del model  # remove to demonstrate saving and loading
    #
    # model = PPO.load("ppo_llvm")


    print('===========Evaluating============')
    obs = env.reset()
    step = 0
    dones = False
    cum_reward = 0
    while not dones:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        print(f'Step {step}: Reward = {rewards}')
        cum_reward += rewards
        step += 1
    print(f'Cumulative Reward = {cum_reward}')