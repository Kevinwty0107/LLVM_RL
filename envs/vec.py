from typing import List, Optional

from envs.env import LLVMPassEnvironment
from programs.program import Program


class LLVMPassVecEnvironment(LLVMPassEnvironment):
    def __init__(self, programs: List[Program], possible_actions: List,
                 state_encoding='instcount', history_length=-1,
                 observe_runtime=True, normalization_metric: str = 'O0-O3',
                 repeat_runs: Optional[List[int]] = None):
        super().__init__(programs[0], possible_actions, state_encoding,
                         history_length, observe_runtime, normalization_metric)

        for prog in programs[1:]:
            _, _, has_exe = prog.check_build('clean_build')
            if not has_exe:
                prog.build_executable('clean_build')
            prog.copy_build('clean_build', 'build')
            prog.calibrate_baseline()

        self._prog_list = programs
        if repeat_runs:
            assert len(repeat_runs) == len(programs), 'repeat_runs must have the same shape as programs'
            self._run_order = self._gen_run_order(repeat_runs)
        else:
            self._run_order = list(range(len(programs)))

    def reset(self, return_info: bool = False):
        self._run_order = self._run_order[1:] + [self._run_order[0]]
        self._program = self._prog_list[self._run_order[0]]
        return super().reset(return_info)

    def _gen_run_order(self, repeat_runs):
        run_order = []
        for i in range(len(repeat_runs)):
            run_order.extend([i] * repeat_runs[i])
        return run_order

