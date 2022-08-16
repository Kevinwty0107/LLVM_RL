import pickle

import gym
import matplotlib.pyplot as plt
from gym import Env

from benchmarks.coremark import CoreMarkProgram
from config.O3_passes import VALID_O3_PASSES
from config.config import COREMARK_LOC
from envs.env import LLVMPassEnvironment


def explore_random(env: Env, seed=1, fig=None):
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    _, info = env.reset(return_info=True)
    scores = [info['norm_run_score'][0]]

    done = False
    while not done:
        action = env.action_space.sample()
        ob, reward, done, info = env.step(action)
        scores.append((action, ob, reward, done, info))

    with open(f'random/seed_{seed}', 'wb+') as f:
        pickle.dump(scores, f)


if __name__=='__main__':
    prog = CoreMarkProgram(COREMARK_LOC)
    env = LLVMPassEnvironment(prog, VALID_O3_PASSES)
    env = gym.wrappers.TimeLimit(env, 200)
    for seed in range(5, 10):
        explore_random(env, seed)
    # with open(f'random/seed_1', 'rb') as f:
    #     b = pickle.load(f)
    #     print(b)