import pickle

import numpy as np
from matplotlib import pyplot as plt


def plot_ppo():
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))

    for i in range(3):
        n_steps = [320, 1024, 2048][i]
        for seed in [1, 2, 3, 4]:
            print(f'=========Evaluating: {n_steps}-{seed}==========')
            with open(f'logs/eval/ppo_scorelog_{n_steps}_{seed}', 'rb+') as f:
                actions, scores = pickle.load(f)

            ax[i].plot(scores, label=f'PPO, seed={seed}')

        ax[i].set_xlabel('Iterations')
        ax[i].set_ylabel('Score (O0-O3 normalized)')
        ax[i].set_title(f'PPO Agent, n_steps={n_steps}')
        ax[i].set_ylim(-0.7, 1.6)
        ax[i].legend(loc='lower right')

    plt.savefig(f'PPO.png', bbox_inches='tight')
    plt.close()


def plot_bar():
    random_scores = []
    greedy_scores = []
    ppo_scores = []
    for seed in range(1, 30):
        with open(f'../baselines/random/seed_{seed}', 'rb') as f:
            l = pickle.load(f)
            random_scores.append(l[200][2])

    for seed in range(1, 4):
        with open(f'../baselines/greedy/seed_{seed}', 'rb') as f:
            scores, actions = pickle.load(f)
            greedy_scores.append(scores[-1])

    for seed in range(1, 9):
        with open(f'logs/eval/ppo_scorelog_2048_{seed}', 'rb+') as f:
            actions, scores = pickle.load(f)
            ppo_scores.append(scores[-1])

    plt.bar(range(4), (np.mean(random_scores), np.mean(greedy_scores), np.mean(ppo_scores), 1),
            yerr=(np.std(random_scores), np.std(greedy_scores), np.std(ppo_scores), 0))
    plt.xticks(range(4), ['Random', 'Greedy', 'PPO', 'O3'])
    plt.savefig('bar.png')


if __name__ == "__main__":
    plot_bar()
