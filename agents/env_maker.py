import gym
import numpy as np

from benchmarks.coremark import CoreMarkProgram
from benchmarks.himeno import HimenoProgram
from benchmarks.polybench import PolyBenchProgram
from config.O3_passes import VALID_O3_PASSES
from config.config import COREMARK_LOC, POLYBENCH_LOC, POLYBENCH_PROGS_BY_TIME, HIMENO_LOC
from envs.env import LLVMPassEnvironment
from envs.vec import LLVMPassVecEnvironment
from envs.wrappers import TerminalAction, RewardDifference, RecordRollouts, DummyTerminalAction
from util.util import denoise, flatten


def make_env(args):
    if args.bench_name == 'polybench':
        programs = [PolyBenchProgram(POLYBENCH_LOC, p) for p in flatten(POLYBENCH_PROGS_BY_TIME[:-3])]
        env = LLVMPassVecEnvironment(programs, VALID_O3_PASSES, args.encoding, args.action_history, False)
    elif args.bench_name == "himeno":
        program = HimenoProgram(HIMENO_LOC)
        env = LLVMPassEnvironment(program, VALID_O3_PASSES, args.encoding, args.action_history, False)
    else:
        program = CoreMarkProgram(COREMARK_LOC)
        env = LLVMPassEnvironment(program, VALID_O3_PASSES, args.encoding, args.action_history, False)

    if args.episode_timesteps > 0:
        env = gym.wrappers.TimeLimit(env, args.episode_timesteps)

    transform = None
    if args.reward_scaling == 'exp':
        transform = lambda x: 2 ** (min(5 * x, 10))
    elif args.reward_scaling == 'cubic':
        transform = lambda x: x ** 3
    elif args.reward_scaling == 'log':
        transform = lambda x: 100 * np.log2(x)
    if transform:
        env = gym.wrappers.TransformReward(env, transform)

    if args.record_rollouts:
        env = RecordRollouts(env, args.record_rollouts)

    if args.use_reward_difference:
        env = RewardDifference(env, lambda _, info: transform(
            np.mean(info['norm_run_score'])) if transform else
            np.mean(info['norm_run_score']))

    if args.use_terminal_action:
        env = DummyTerminalAction(env)

    if args.denoise_threshold > 0:
        env = gym.wrappers.TransformReward(env, lambda x: denoise(x, args.denoise_threshold))

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    return env


# def make_thunk(args, prog_name):
#     def thunk():
#         program = PolyBenchProgram(POLYBENCH_LOC, prog_name)
#         env = LLVMPassEnvironment(program, VALID_O3_PASSES, args.encoding, args.action_history, False)
#
#         transform = None
#         if args.reward_scaling == 'exp':
#             transform = lambda x: 2 ** (min(5 * x, 10))
#         elif args.reward_scaling == 'cubic':
#             transform = lambda x: x ** 3
#         elif args.reward_scaling == 'log':
#             transform = lambda x: 100 * np.log2(x)
#         if transform:
#             env = gym.wrappers.TransformReward(env, transform)
#
#         if args.use_reward_difference:
#             env = RewardDifference(env, lambda _, info: transform(
#                 np.mean(info['norm_run_score'])) if transform else
#                 np.mean(info['norm_run_score']))
#
#         if args.use_terminal_action:
#             env = TerminalAction(env)
#
#         if args.denoise_threshold > 0:
#             env = gym.wrappers.TransformReward(env, lambda x: denoise(x, args.denoise_threshold))
#
#         if args.episode_timesteps > 0:
#             env = gym.wrappers.TimeLimit(env, args.episode_timesteps)
#
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         env.seed(args.seed)
#         env.action_space.seed(args.seed)
#         env.observation_space.seed(args.seed)
#         return env
#
#     return thunk
