# Size of minibatch
import time
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from agents.world_model.load_rollouts import load_rollouts
from agents.world_model.mdnrnn import MDRNN, gmm_loss

BSIZE = 10
# Size of Latent
LSIZE = 340
# Size of hidden
HSIZE = 256

# Number of Gaussians
N_GAUSS = 5
# Length of action sequence
SEQ_LEN = 200

mdrnn = MDRNN(LSIZE, 1, HSIZE, N_GAUSS)

optimizer = torch.optim.Adam(mdrnn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

def get_loss(obs, actions, rewards, next_obs):
    obs, actions, rewards, next_obs = [
        arr.transpose(1, 0) for arr in [
            obs, actions, rewards, next_obs]]

    mus, sigmas, logpi, rs = mdrnn(actions, obs)
    gmm = gmm_loss(next_obs, mus, sigmas, logpi)
    mse = F.mse_loss(rs, rewards) * 100
    scale = LSIZE + 1
    loss = (gmm + mse) / scale
    return dict(gmm=gmm, mse=mse, loss=loss)


def train(dataloader: DataLoader, epochs=40, logger: Optional[SummaryWriter] = None):
    for epoch in range(epochs):
        epoch_loss = []
        epoch_gmm = []
        epoch_mse = []

        for data in dataloader:
            obs, next_obs, actions, rewards = data
            losses = get_loss(obs, actions, rewards, next_obs)
            optimizer.zero_grad()
            losses['loss'].backward()
            optimizer.step()

            epoch_loss.append(losses['loss'])
            epoch_gmm.append(losses['gmm'])
            epoch_mse.append(losses['mse'])

        avg_loss = torch.mean(Tensor(epoch_loss))
        avg_gmm = torch.mean(Tensor(epoch_gmm))
        avg_mse = torch.mean(Tensor(epoch_mse))

        scheduler.step(avg_loss)

        print(f'Epoch {epoch}:\n'
              f'  LOSS - {avg_loss}\n'
              f'  GMM - {avg_gmm}\n'
              f'  MSE - {avg_mse}\n')

        if logger:
            logger.add_scalar('losses/loss', avg_loss, epoch)
            logger.add_scalar('losses/gmm', avg_gmm, epoch)
            logger.add_scalar('losses/mse', avg_mse, epoch)

if __name__ == '__main__':
    rollout_obs, rollout_next_obs, rollout_actions, rollout_rewards = load_rollouts('data/rollouts')
    rollout_actions = torch.unsqueeze(rollout_actions, -1)
    dataset = TensorDataset(rollout_obs, rollout_next_obs, rollout_actions, rollout_rewards)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    logger = SummaryWriter(f'logs/{time.time()}')
    train(loader, logger=logger)
    torch.save(mdrnn.state_dict(), 'trained/mdrnn')



