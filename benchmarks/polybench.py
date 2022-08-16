import os
from subprocess import check_output, Popen

import numpy as np

from programs.program import Program
from config.config import POLYBENCH_LM


class PolyBenchProgram(Program):
    _file_name = None
    _build_bc_command = 'clang {opt_flags} -flto -fuse-ld=ld -Wl,-plugin-opt=emit-llvm ' \
                        '-I ../utilities -I ../{file_loc}/{file_name} ../utilities/polybench.c' \
                        ' ../{file_loc}/{file_name}/{file_name}.c -DPOLYBENCH_TIME -o {file_name}.bc'
    _repeat_per_measure = 1
    _parallel_measurement = False

    _name_to_loc = {
        'gemm': 'linear-algebra/blas',
        'gemver': 'linear-algebra/blas',
        'gesummv': 'linear-algebra/blas',
        'symm': 'linear-algebra/blas',
        'syr2k': 'linear-algebra/blas',
        'syrk': 'linear-algebra/blas',
        'trmm': 'linear-algebra/blas',
        '2mm': 'linear-algebra/kernels',
        '3mm': 'linear-algebra/kernels',
        'atax': 'linear-algebra/kernels',
        'bicg': 'linear-algebra/kernels',
        'doitgen': 'linear-algebra/kernels',
        'mvt': 'linear-algebra/kernels',
        'cholesky': 'linear-algebra/solvers',
        'durbin': 'linear-algebra/solvers',
        'gramschmidt': 'linear-algebra/solvers',
        'lu': 'linear-algebra/solvers',
        'ludcmp': 'linear-algebra/solvers',
        'trisolv': 'linear-algebra/solvers',
        'correlation': 'datamining',
        'covariance': 'datamining',
        'deriche': 'medley',
        'floyd-warshall': 'medley',
        'nussinov': 'medley',
        'adi': 'stencils',
        'fdtd-2d': 'stencils',
        'heat-3d': 'stencils',
        'jacobi-1d': 'stencils',
        'jacobi-2d': 'stencils',
        'seidel-2d': 'stencils'
    }

    def __init__(self, root_dir: str, file_name: str, build: str = None):
        super().__init__(root_dir, build)
        self._file_name = file_name
        if file_name in POLYBENCH_LM:
            self._build_exe_command = 'clang {file_name}.bc -o {file_name}1 -lm && mv {file_name}1 {file_name}'

    def build_bc(self, build: str = 'build', opt_lvl='-O0'):
        cwd = self.chdir_build(build)
        print('Building bitcode ...')
        opt_flags = opt_lvl
        if opt_lvl == '-O0':
            opt_flags += ' -Xclang -disable-O0-optnone'

        os.system(self._build_bc_command.format(file_name=self._file_name,
                                                file_loc=self._name_to_loc[self._file_name],
                                                opt_flags=opt_flags))
        os.chdir(cwd)

    def measure_run_time(self, build: str = 'build', repeat=1, parallel=False) -> np.ndarray:
        binary_path = os.path.join(self.root_dir, build, self._file_name)
        if not os.path.isfile(binary_path):
            raise ValueError("Binary not found: {}".format(binary_path))

        run_times = []

        for workload in self._workloads:
            command = '{} {}'.format(binary_path, self._run_args.format(workload=workload))
            if parallel:
                procs = [Popen(command, shell=True) for _ in range(repeat)]
                run_time = np.mean([float(proc.communicate()[0].decode('utf-8')) for proc in procs])
            else:
                run_time = np.mean([float(check_output(command, shell=True)) for _ in range(repeat)])
            run_times.append(run_time)
        return np.array(run_times)