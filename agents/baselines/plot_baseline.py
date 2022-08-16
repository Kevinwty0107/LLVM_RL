import pickle

import numpy as np
from matplotlib import pyplot as plt

from config.O3_passes import O3_PASS_SEQUENCE


def plot_random():
    for seed in range(1, 5):
        with open(f'random/seed_{seed}', 'rb') as f:
            l = pickle.load(f)
            print(l[-1][3:])
            scores = [x[2] for x in l[1:200]]
            scores.insert(0, l[0])
            actions = [x[0] for x in l[1:200]]
            actions.insert(0, 0)
            print(scores)
            plt.plot(scores, label=f"random agent, seed={seed}")
    plt.xlabel("Iterations")
    plt.ylabel("Score (00-O3 normalized)")
    plt.axhline(0, color='indianred', ls='--', label='O0')
    plt.axhline(1, color='royalblue', ls='--', label='O3')
    plt.legend()
    plt.title('Random Agent Performance on CoreMark')
    plt.savefig("random_agent_1.png")
    plt.close()

def plot_greedy():
    for seed in range(1, 4):
        with open(f'greedy/seed_{seed}', 'rb') as f:
            scores, actions = pickle.load(f)
            plt.plot(scores, label=f"greedy agent, seed={seed}")
    plt.xlabel("Iterations")
    plt.ylabel("Score (00-O3 normalized)")
    plt.ylim((-0.1, 1.1))
    plt.axhline(0, color='indianred', ls='--', label='O0')
    plt.axhline(1, color='royalblue', ls='--', label='O3')
    plt.legend()
    plt.title('Greedy Agent Performance on CoreMark')
    plt.savefig("greedy_agent_1.png")
    plt.close()

def plot_o3():
    with open('o3/o3', 'rb') as f:
        scores = pickle.load(f)
        scores = [score/scores[-1] for score in scores]
        actions = O3_PASS_SEQUENCE

        plt.plot(scores, label='O3 agent')
        plt.xlabel("Iterations")
        plt.ylabel("Score (00-O3 normalized)")
        plt.axhline(0, color='indianred', ls='--', label='O0')
        plt.axhline(1, color='royalblue', ls='--', label='O3')
        plt.legend()
        plt.title('O3 Agent Performance on CoreMark')
        plt.savefig("o3_agent_1.png")
        plt.close()

        plt.plot(scores)
        plt.xlabel("Iterations")
        plt.ylabel("Score (00-O3 normalized)")
        simplifycfg = [i for i, a in enumerate(actions) if a == 'simplifycfg']
        plt.axvline(simplifycfg[0], color='royalblue', ls='--', label='simplifycfg')
        for x in simplifycfg[1:]:
            plt.axvline(x, color='royalblue', ls='--')
        plt.title('O3 Agent Performance on CoreMark')
        plt.legend()
        plt.savefig("o3_agent_2.png")
        plt.close()

if __name__ == "__main__":
    # plot_random()
    # plot_greedy()
    # plot_o3()
    plot_barchart()