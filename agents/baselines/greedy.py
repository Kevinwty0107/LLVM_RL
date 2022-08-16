import os
import pickle
import random

import gym
import numpy as np

from agents.baselines.random import explore_random
from benchmarks.coremark import CoreMarkProgram
from config.O3_passes import VALID_O3_PASSES
from config.config import COREMARK_LOC
from envs.env import LLVMPassEnvironment
from programs.program import Program


def explore_greedy_iter(prog: Program, k=20):
    prog.chdir_build()
    os.system('cp coremark.bc clean.bc')
    best_score = -np.inf
    best_action = None
    actions = random.sample(VALID_O3_PASSES, k=k)
    for action in actions:
        prog.opt(f'-{action}')

        prog.build_executable()
        score = prog.measure_normalized_run_score()

        os.system('cp clean.bc coremark.bc')
        if score > best_score:
            best_score = score
            best_action = action
            os.system('cp coremark.bc best.bc')

    os.system('cp best.bc coremark.bc')
    print(f'=========BEST SCORE: {best_score}============')
    return best_score, best_action


def explore_greedy(prog: Program, steps=200):
    scores = []
    actions = []
    for _ in range(steps):
        score, action = explore_greedy_iter(prog)
        scores.append(score)
        actions.append(action)
    return scores, actions


if __name__=="__main__":
    prog = CoreMarkProgram(COREMARK_LOC)
    _, _, has_exe = prog.check_build('clean_build')
    if not has_exe:
        prog.build_executable('clean_build')
    prog.copy_build('clean_build', 'build')
    for i in range(1, 5):
        random.seed(i)
        prog.calibrate_baseline()
        scores, actions = explore_greedy(prog)
        os.chdir(os.path.dirname(__file__))
        with open(f'greedy/seed_{i}_new', 'wb+') as f:
            pickle.dump((scores, actions), f)
