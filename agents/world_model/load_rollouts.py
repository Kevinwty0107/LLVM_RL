import pickle

import numpy as np
import torch
from torch import Tensor


def load_rollouts(path):
    f = open(path, 'rb')
    rollout_obs = torch.zeros(0, 200, 340)
    rollout_next_obs = torch.zeros(0, 200, 340)
    rollout_actions = torch.zeros(0, 200)
    rollout_rewards = torch.zeros(0, 200)
    while True:
        try:
            obs, next_obs, actions, rewards = list(map(
                lambda x: Tensor(np.array(x)).unsqueeze(0), pickle.load(f)))
            rollout_obs = torch.cat((rollout_obs, obs))
            rollout_next_obs = torch.cat((rollout_next_obs, next_obs))
            rollout_actions = torch.cat((rollout_actions, actions))
            rollout_rewards = torch.cat((rollout_rewards, rewards))
        except (EOFError, pickle.UnpicklingError):
            break

    return rollout_obs, rollout_next_obs, rollout_actions, rollout_rewards

if __name__ == "__main__":
    load_rollouts('data/rollouts_diff')
