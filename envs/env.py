import math
from typing import List, Optional

from gym.utils import seeding

import numpy as np
from gym import Env, spaces
from numpy import float32

from programs.program import Program
from config.config import COUNTED_INST


class LLVMPassEnvironment(Env):
    def __init__(self, program: Program, possible_actions: List,
                 state_encoding='instcount', history_length=-1,
                 observe_runtime=True, normalization_metric: str = 'O0-O3'):
        # Action Space
        self.action_space = spaces.Discrete(len(possible_actions))

        # Observation Space
        low = []
        high = []
        shape = 0

        if state_encoding == 'instcount':
            low += [0.] * len(COUNTED_INST)
            high += [1e9] * len(COUNTED_INST)
            shape += len(COUNTED_INST)
        elif state_encoding == 'ir2vec':
            low += [-np.inf] * 300
            high += [np.inf] * 300
            shape += 300

        if history_length >= 0:
            low += [0.] * history_length
            high += [self.action_space.n] * history_length
            shape += history_length
        else:
            low += [0.] * self.action_space.n
            high += [100.] * self.action_space.n
            shape += self.action_space.n

        if observe_runtime:
            low += [-10]
            high += [10]
            shape += 1

        low = np.array(low, dtype=float32)
        high = np.array(high, dtype=float32)
        shape = (shape,)

        self.observation_space = spaces.Box(low, high, shape)

        # Other attributes
        self._program = program
        _, _, has_exe = self._program.check_build('clean_build')
        if not has_exe:
            self._program.build_executable('clean_build')
        self._program.copy_build('clean_build', 'build')
        self._program.calibrate_baseline()
        self._metric = normalization_metric

        # self._measure_each_step = False

        # This should be the full list of possible passes (with arguments)
        # Currently, arguments are not supported
        # Maybe use PASS_DATA to generate passes?
        # self._passes = [data[0] for data in PASS_DATA.values()]

        # In this iteration, we use predefined passes with predefined arguments.
        self._passes = possible_actions

        self._history_length = history_length
        self._state_encoding = state_encoding
        self._observe_runtime = observe_runtime

        self._history_passes = None
        self._code_state = None
        self._code_vec = None
        # self._last_build_time = None
        self._last_binary_size = None
        self._last_run_score = None
        self._last_norm_run_score = None

    def _gen_observation(self):
        observation = []

        if self._state_encoding == 'instcount':
            for inst in COUNTED_INST:
                if inst in self._code_state:
                    observation = np.append(observation, self._code_state[inst])
                else:
                    observation = np.append(observation, 0)
        elif self._state_encoding == 'ir2vec':
            observation = np.concatenate((observation, self._code_vec))

        padded_history = []
        if self._history_length >= 0:
            pad_len = self._history_length - len(self._history_passes)
            if pad_len > 0:
                padded_history = np.pad(self._history_passes, (pad_len, 0))
            elif self._history_length > 0:
                padded_history = self._history_passes[-self._history_length:]
        else:
            padded_history = np.bincount(self._history_passes, minlength=self.action_space.n)
        observation = np.concatenate((observation, padded_history))

        if self._observe_runtime:
            observation = np.append(observation, self._last_norm_run_score)

        return observation

    def _gen_info(self):
        return {"history_passes": self._history_passes,
                "code_state": self._code_state,
                "binary_size": self._last_binary_size,
                "run_score": self._last_run_score,
                "norm_run_score": self._last_norm_run_score}

    def reset(self, return_info: bool = False):
        self._history_passes = []

        # Build without optimization on reset
        _, _, has_exe = self._program.check_build('clean_build')
        if not has_exe:
            self._program.build_executable('clean_build')
        self._program.copy_build('clean_build', 'build')
        # self._program.calibrate_baseline()
        # self._last_build_time = 0.0
        self._last_binary_size = self._program.measure_binary_size()
        self._last_run_score = self._program.measure_run_score()
        self._last_norm_run_score = self._program.normalize_run_score(self._last_run_score, metric=self._metric)

        # Analyze the program
        self._code_state = self._program.instcount()
        if self._state_encoding == "ir2vec":
            self._code_vec = self._program.ir2vec()

        if return_info:
            return self._gen_observation(), self._gen_info()
        else:
            return self._gen_observation()

    def step(self, action: int):
        # Get pass to run
        pass_id = action
        pass_name = self._passes[pass_id]

        self._history_passes.append(pass_id)

        build_time = self._program.opt(f'-{pass_name}')
        if build_time < 0:
            reward = 0
        else:
            ret_code = self._program.build_executable()
            if ret_code != 0:
                reward = 0
            else:
                self._last_binary_size = self._program.measure_binary_size()
                self._last_run_score = self._program.measure_run_score()
                self._last_norm_run_score = self._program.normalize_run_score(self._last_run_score, metric=self._metric)
                reward = np.mean(self._last_norm_run_score)

                # Analyze the program
                self._code_state = self._program.instcount()
                if self._state_encoding == "ir2vec":
                    self._code_vec = self._program.ir2vec()

        return self._gen_observation(), reward, False, self._gen_info()

    def render(self, mode="human"):
        pass
