import os
from typing import List

from util.measure import measure_run_time
from programs.program import Program


class Project(Program):
    # Executed in build
    _build_bc_command = 'CC=clang CXX=clang++ cmake -DOptFlags="{opt_flags}" LDFLAGS="-flto -fuse-ld=ld -Wl,-plugin-opt=save-temps" ..;' \
                        'cmake --build . -- -j8;' \
                        'mv {file_name} {file_name}.bc'
    _build_exe_command = 'clang {file_name}.bc -o {file_name} -static -lstdc++ -lpthread -lm'
    _clean_command = 'rm -rf *; rm -rf .*'

    _file_name = None  # Type: str
    _run_args = None  # Type: str
    _workloads = []  # Type: List[str]

    def __init__(self, root_dir: str, build: str = None):
        super().__init__(root_dir, build)

    def measure_run_time(self, build: str = 'build', verbose=False) -> List[float]:
        binary_path = os.path.join(self.root_dir, build, self._file_name)
        if not os.path.isfile(binary_path):
            raise ValueError("Binary not found: {}".format(binary_path))

        run_times = []

        for workload in self._workloads:
            abs_workload = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', workload))
            command = f'{binary_path} {self._run_args.format(workload=abs_workload)}'
            run_time = measure_run_time(command, verbose=verbose)
            # print(command, run_time)
            run_times.append(run_time)

        return run_times

    # def measure_build_time(self, build: str = 'build'):
    #     build = os.path.join(self.root_dir, build)
    #
    #     cwd = os.getcwd()
    #     if not os.path.exists(build):
    #         os.mkdir(build, 0o755)
    #     os.chdir(build)
    #     os.system(self._clean_command)
    #
    #     start_time = time.perf_counter()
    #     self.build_executable(build)
    #     time_taken = time.perf_counter() - start_time
    #
    #     os.chdir(cwd)
    #
    #     return time_taken
