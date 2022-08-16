import os
import unittest

from llvm_rl.agents.random_agent import RandomAgent
from llvm_rl.env import LLVMPassEnvironment
from llvm_rl.predef_pass_lists import PredefPassList
from llvm_rl.programs.lepton import LeptonProgram


class TestEnvLepton(unittest.TestCase):
    def setUp(self) -> None:
        src_dir = os.path.join(
            os.path.dirname(__file__),
            '..',
            '..',
            'programs',
            'lepton.bc'
        )

        build_dir = os.path.join(
            src_dir,
            'build'
        )

        self.program = LeptonProgram(
            src_dir=src_dir,
            build_dir=build_dir
        )

        self.env = LLVMPassEnvironment(self.program, PredefPassList.O3_LIST)

    def test_train(self):
        # self.skipTest("Skip step.")
        agent = RandomAgent(self.env.num_actions)

        # Reset environment. This will usually trigger a build
        state = self.env.reset()

        for step in range(100):
            # Call model here to return action. In this example,
            # a random action is chosen.
            action = agent.get_action(state)

            # If action=0 was chosen, the environment will build the program
            next_state, reward, done, info = self.env.step(action)

            if action == 0:
                assert done
                pass_list, build_time, binary_size, run_time = next_state
                assert build_time > 10.0
                assert binary_size > 0
                assert run_time > 0.5

            # Send result to agent
            agent.observe(state, action, reward, done, next_state)

            if done:
                # Potentially update model here
                agent.update()

                # Reset environment
                next_state = self.env.reset()

            state = next_state


if __name__ == '__main__':
    unittest.main()
