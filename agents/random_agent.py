import random
from typing import List


class RandomAgent(object):
    def __init__(self, num_passes: int):
        self._num_passes = num_passes

    def get_action(self, state):
        pass_list, build_time, binary_size, run_time, stats = state
        # We do not use the state in a random agent

        # A non-random agent should call a model here
        return random.randint(0, self._num_passes-1)

    def observe(self, state, action, reward, done, next_state):
        # A non-random agent should store the experiences here
        pass

    def update(self):
        # A non-random agent should update the model here
        pass
