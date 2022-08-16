from llvm_rl_simp.env import LLVMPassEnvironment
from llvm_rl_simp.agents.random_agent import RandomAgent
from llvm_rl_simp.programs.program import Program
from llvm_rl_simp.valid_passes import PASS_ALIAS
from llvm_rl_simp.measure import measure_run_time

from matplotlib import pyplot as plt

class Bubblesort(Program):
    _file_name = 'Bubblesort'
    _run_args = ''
    _workloads = ['']


program = Bubblesort(src_dir='/mnt/d/Cambridge/DissProjectM/llvm/stanford')
O3_time = measure_run_time('/mnt/d/Cambridge/DissProjectM/llvm/stanford/Bubblesort_clang')
env = LLVMPassEnvironment(program=program, possible_actions=PASS_ALIAS)
reward_history = []
agent = RandomAgent(env.num_actions)
state = env.reset()
done = False
for _ in range(100):
    action = agent.get_action(state)
    result = env.step(int(action))
    print(result)
    state, reward, done, _ = result
    reward_history.append(reward)
    if done:
        break

plt.plot(reward_history)
plt.axhline(y=O3_time, color='r')
plt.savefig('random_agent.png')