import argparse
import os
import random
import time
from typing import Literal

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from benchmarks.coremark import CoreMarkProgram, COREMARK_LOC
from config.O3_pass_groups import O3_PASS_GROUPS_AS_PASSES
from config.config import DQN_STATE_DICT_LOC, DQN_TARGET_STATE_DICT_LOC
from config.O3_passes import VALID_O3_PASSES
from envs.env import LLVMPassEnvironment
from envs.wrappers.reward_difference import RewardDifference
from envs.wrappers.terminal_action import TerminalAction
from envs.wrappers.time_limit_with_ob import TimeLimitWithOb


def parse_args():
    # Exp Related
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="name of experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of experiment")

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

    # Agent Related
    parser.add_argument("--total-timesteps", type=int, default=20000,
                        help="total timesteps of experiment")
    parser.add_argument("--learning-rate", type=float, default=0.00025,
                        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=12000,
                        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--target-update-frequency", type=int, default=500,
                        help="timesteps to update the target network")
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
    parser.add_argument("--train-frequency", type=int, default=1,
                        help="the frequency of training")
    args = parser.parse_args()

    return args


def denoise(x, denoise_thres):
    y = 100 * x
    if abs(y) > denoise_thres:
        return y
    else:
        return (y / denoise_thres) ** 3 * denoise_thres


def make_env(args):
    program = CoreMarkProgram(COREMARK_LOC)
    env = LLVMPassEnvironment(program, O3_PASS_GROUPS_AS_PASSES)

    if args.use_terminal_action:
        env = TerminalAction(env)

    transform = None
    if args.reward_scaling == 'exp':
        transform = lambda x: 2 ** (min(5 * x, 10))
    elif args.reward_scaling == 'cubic':
        transform = lambda x: x ** 3
    elif args.reward_scaling == 'log':
        transform = lambda x: 100 * np.log2(x)
    if transform:
        env = gym.wrappers.TransformReward(env, transform)

    if args.use_reward_difference:
        env = RewardDifference(env, lambda _, info: transform(
            np.mean(info['norm_run_score']) if transform else
            np.mean(info['norm_run_score'])))

    if args.denoise_threshold > 0:
        env = gym.wrappers.TransformReward(env, lambda x: denoise(x, args.denoise_threshold))

    if args.episode_timesteps > 0:
        env = TimeLimitWithOb(env, args.episode_timesteps)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    return env


# DQN Model:
class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 128),
            nn.ReLU(),
            # nn.Linear(128, 256),
            # nn.ReLU(),
            # nn.Linear(256, 512),
            # nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x.float())


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # envs setup
    env = make_env(args)

    q_network = QNetwork(env).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(env).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size, env.observation_space, env.action_space, device=device,
        optimize_memory_usage=True
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    ob = env.reset()
    cum_reward = 0
    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps,
                                  global_step)
        # if reward < 6600:
        #     print(f'Reward too low ({reward}), attempting to reset')
        #     reward = 7000
        #     action = 0
        if cum_reward < -8 and random.random() < epsilon:
            action = VALID_O3_PASSES.index('simplifycfg')
            cum_reward = 0
        elif random.random() < epsilon:
            action = env.action_space.sample()
        else:
            # print(ob)
            logit = q_network(torch.Tensor(ob).to(device))
            # print(logit)
            action = torch.argmax(logit).cpu().numpy()

        # execute the game and log data.
        next_ob, reward, done, info = env.step(action)
        cum_reward += reward
        cum_reward = min(cum_reward, 0)
        print(f'Reward = {reward}')

        # record rewards for plotting purposes
        if "episode" in info.keys():
            print('Writing summary (episode)...')
            print(f'Current Location is {os.getcwd()}')
            print(f'global_step={global_step}, episodic_return={info["episode"]["r"]}')
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)
            # writer.add_scalar("charts/runtime", info["run_time"], global_step)
            with open('../dqn_info.txt', 'a+') as f:
                f.write(f'{info["episode"]["r"]}\n')

        # save data to reply buffer; handle `terminal_observation`
        real_next_ob = next_ob.copy()
        rb.add(ob, real_next_ob, np.array([action]), np.array([reward]), np.array([done]), [info])
        if done:
            if info.get('TimeLimit.truncated'):
                print('Running past episode limit, reset')
            next_ob = env.reset()
            cum_reward = 0
        ob = next_ob

        # training.
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            if global_step % 50 == 0:
                print('Writing summary (training)...')
                print(f'Current Location is {os.getcwd()}')
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(q_network.parameters()), args.max_grad_norm)
            optimizer.step()

            # torch.save(q_network.state_dict(), DQN_STATE_DICT_LOC)

            # update the target network
            if global_step % args.target_update_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())
                torch.save(target_network.state_dict(), DQN_TARGET_STATE_DICT_LOC)

    env.close()
    writer.close()
