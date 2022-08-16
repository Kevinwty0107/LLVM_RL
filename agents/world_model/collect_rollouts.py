import pickle
from typing import Optional

import numpy as np
from gym.wrappers import TimeLimit
from stable_baselines3.common.base_class import BaseAlgorithm

from benchmarks.coremark import CoreMarkProgram
from benchmarks.himeno import HimenoProgram
from benchmarks.polybench import PolyBenchProgram
from config.O3_passes import VALID_O3_PASSES
from config.config import HIMENO_LOC, COREMARK_LOC, POLYBENCH_LOC, POLYBENCH_PROGS_BY_TIME
from envs.vec import LLVMPassVecEnvironment
from envs.wrappers import TerminalAction, RewardDifference
from util.util import flatten


def collect_rollouts(env, model: Optional[BaseAlgorithm], rollout_nums):
    collected = 0

    while collected < rollout_nums:
        obs = []
        next_obs = []
        actions = []
        rewards = []

        ob = env.reset()
        done = False

        while not done:
            if model:
                action, _states = model.predict(ob)
            else:
                action = env.action_space.sample()
            if action == 0:
                action = VALID_O3_PASSES.index('simplifycfg') + 1
            next_ob, reward, done, _ = env.step(action)
            obs.append(ob)
            next_obs.append(next_ob)
            actions.append(action)
            rewards.append(reward)

            ob = next_ob

        with open('data/rollout_himeno', 'ab+') as f:
            pickle.dump((obs, next_obs, actions, rewards), f)

        print(f'==========ROLLOUT {collected} COLLECTED===========')
        # print(f'| Program Used:  {env._program._file_name: =30}  |')
        # print(f'==================================================')
        collected += 1


if __name__ == "__main__":
    himeno = HimenoProgram(HIMENO_LOC)
    himeno_repeat = 5
    coremark = CoreMarkProgram(COREMARK_LOC)
    coremark_repeat = 5
    polybench = [[PolyBenchProgram(POLYBENCH_LOC, p) for p in ps] for ps in POLYBENCH_PROGS_BY_TIME[:2]]
    poly_repeat = [4, 2]
    polybench_repeat = [[poly_repeat[i] for p in POLYBENCH_PROGS_BY_TIME[i]] for i in range(2)]

    progs = [himeno]
    repeats = None

    env = LLVMPassVecEnvironment(progs, VALID_O3_PASSES, 'ir2vec',
                                 history_length=40, observe_runtime=False)
    # env = RewardDifference(env, lambda _, info: np.mean(info['norm_run_score']))
    env = TerminalAction(env)
    env = TimeLimit(env, 200)

    collect_rollouts(env, None, 300)