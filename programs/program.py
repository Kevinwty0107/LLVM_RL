import os
import re
from subprocess import run, check_output, STDOUT, DEVNULL, Popen, TimeoutExpired, PIPE
import time
from typing import Dict, Literal

import numpy as np

from config.config import VOCAB_LOC
from util.measure import measure_binary_size, measure_run_time


class Program(object):
    # Executed in build
    _build_bc_command = 'clang -c {opt_flags} ../{file_name}.c -emit-llvm -o {file_name}.bc'
    _build_exe_command = 'clang {file_name}.bc -o {file_name}1 && mv {file_name}1 {file_name}'
    _clean_command = 'rm -rf *'

    _opt_command = 'rm {file_name}1.bc;' \
                   'opt {pass_name} {file_name}.bc -o {file_name}1.bc &&' \
                   'mv {file_name}1.bc {file_name}.bc'
    _instcount_command = 'opt -enable-new-pm=0 -stats -instcount {file_name}.bc -o {file_name}.bc'
    _ir2vec_command = 'llvm-dis {file_name}.bc -o {file_name}.ll && ' \
                      'ir2vec -fa -vocab {vocab_loc} -o encoding.txt -level p {file_name}.ll'

    _file_name = None  # Type: str
    _run_args = ''  # Type: str
    _workloads = ['']  # Type: List[str]
    _metric_regex = None
    _parallel_measurement = True
    _repeat_per_measure = 1

    def __init__(self, root_dir: str, build: str = None):
        # Root Directory for the program
        self.root_dir = root_dir

        # O0 and O3 run score to normalize run score
        self.O0_score = None
        self.O3_score = None

        # Provides a quick way to setup a program in a folder
        cwd = os.getcwd()
        os.chdir(root_dir)
        if build:
            self.build_bc(build)
        os.chdir(cwd)

    def calibrate_baseline(self, repeat=3):
        # Ensure both clean_build(o0_build) and o3_build has executable
        self.complete_build('clean_build')
        self.copy_build(target_build='o3_build')
        self.opt('-O3', 'o3_build')
        self.build_executable('o3_build')

        self.O0_score = np.mean(
            [self.measure_run_score('clean_build') for _ in range(repeat)])
        self.O3_score = np.mean(
            [self.measure_run_score('o3_build') for _ in range(repeat)])
        print('===========CALIBRATED BASELINE================')
        print(f'O0 : {self.O0_score}')
        print(f'O3 : {self.O3_score}')

    def complete_build(self, build: str = 'build'):
        has_build, has_bc, has_exe = self.check_build(build)
        _, _, has_clean_exe = self.check_build('clean_build')
        if not has_build:
            if not has_clean_exe:
                self.build_executable('clean_build')
            if build != 'clean_build':
                self.copy_build(target_build=build)
        else:
            if not has_exe:
                self.build_executable(build)

    # Returns has_build, has_bc, has_exe
    def check_build(self, build: str = 'build'):
        build_dir = os.path.join(self.root_dir, build)
        if not os.path.exists(build_dir):
            return False, False, False
        elif not os.path.exists(os.path.join(build_dir, f'{self._file_name}.bc')):
            return True, False, False
        elif not os.path.exists(os.path.join(build_dir, self._file_name)):
            return True, True, False
        else:
            return True, True, True

    def cleanup_build(self, build: str = 'build'):
        cwd = self.chdir_build(build)
        print(f'Start cleaning in: {os.getcwd()} ...')
        os.system(self._clean_command)
        print('Finished Cleaning')
        os.chdir(cwd)

    def copy_build(self, source_build: str = 'clean_build', target_build: str = 'build'):
        self.cleanup_build(target_build)
        cwd = os.getcwd()
        os.chdir(self.root_dir)
        print(f'Start copying from {source_build} to {target_build}')
        os.system(f'cp -R {source_build}/* {target_build}')
        print('Finished copying')
        os.chdir(cwd)

    def chdir_build(self, build: str = 'build'):
        """Check whether there is an existing build
        If not, create an empty one and cd into it"""
        build_dir = os.path.join(self.root_dir, build)
        if not os.path.exists(build_dir):
            os.mkdir(build_dir, 0o755)
        cwd = os.getcwd()
        os.chdir(build_dir)
        return cwd

    def remove_build(self, build: str = 'build'):
        build_dir = os.path.join(self.root_dir, build)
        os.system(f'rm -rf {build_dir}')

    # Run from anywhere
    def build_bc(self, build: str = 'build', opt_lvl='-O0'):
        cwd = self.chdir_build(build)
        print('Building bitcode ...')
        opt_flags = opt_lvl
        if opt_lvl == '-O0':
            opt_flags += ' -Xclang -disable-O0-optnone'

        os.system(self._build_bc_command.format(file_name=self._file_name,
                                                opt_flags=opt_flags))
        os.chdir(cwd)

    # Run from anywhere
    def build_executable(self, build: str = 'build'):
        cwd = self.chdir_build(build)
        # If there is no bitcode, build the bitcode first
        if not os.path.exists(f'{self._file_name}.bc'):
            self.build_bc(build)
        print('Building executable ...')
        proc = run(self._build_exe_command.format(file_name=self._file_name), shell=True,
                   stdout=DEVNULL, stderr=DEVNULL)
        os.chdir(cwd)
        return proc.returncode

    def opt(self, pass_name, build: str = 'build', time_cap=20) -> float:
        cwd = self.chdir_build(build)
        # If there is no bitcode, build the bitcode first
        if not os.path.exists(f'{self._file_name}.bc'):
            self.build_bc(build)

        opt_command = self._opt_command.format(pass_name=pass_name, file_name=self._file_name)

        print(f'Running Optimization -{pass_name} ...')
        start_time = time.perf_counter()
        proc = Popen(opt_command, shell=True, stdout=DEVNULL, stderr=DEVNULL)

        try:
            proc.communicate(timeout=time_cap)
            build_time = time.perf_counter() - start_time
            if proc.returncode != 0:
                build_time = -2
                print('failed!')
        except TimeoutExpired:
            # process was killed due to exceeding the alarm
            # or optimization did not go through
            build_time = -1
            print('timeout!')
        finally:
            os.chdir(cwd)

        return build_time

    def instcount(self, build: str = 'build') -> Dict[str, int]:
        cwd = self.chdir_build(build)
        # If there is no bitcode, build the bitcode first
        if not os.path.exists(f'{self._file_name}.bc'):
            self.build_bc(build)

        state = {}
        print('Analysing Bitcode ...')
        instcount_command = self._instcount_command.format(file_name=self._file_name)
        out = check_output(instcount_command, stderr=STDOUT, shell=True)
        lines = out.decode('utf-8').split('\n')
        regex = r"^\s*([0-9]+)\s*([^\s]*)\s*-\s*Number of ([^\s]+).*"

        for line in lines:
            m = re.search(regex, line)
            if m:
                num, tool, inst = m.groups()
                if tool == "instcount":
                    state[inst] = int(num)

        os.chdir(cwd)

        return state

    def ir2vec(self, build: str = 'build'):
        cwd = self.chdir_build(build)
        # If there is no bitcode, build the bitcode first
        if not os.path.exists(f'{self._file_name}.bc'):
            self.build_bc(build)

        # Encode Bitcode
        print('Encoding Bitcode ...')
        ir2vec_command = self._ir2vec_command.format(vocab_loc=VOCAB_LOC, file_name=self._file_name)
        proc = run(ir2vec_command, stdout=DEVNULL, stderr=DEVNULL, shell=True)

        encodings = None
        if proc.returncode == 0:
            # Encoding successful
            with open('encoding.txt') as f:
                encodings = f.readlines()[-1].split()
                # encodings = list(map(int, encodings))
        os.chdir(cwd)

        if encodings is None:
            raise Exception("IR2Vec encoding failed!")

        return np.array(encodings, dtype="float32")

    def measure_binary_size(self, build: str = 'build') -> int:
        binary_path = os.path.join(self.root_dir, build, self._file_name)
        if not os.path.isfile(binary_path):
            raise ValueError("Binary not found: {}".format(binary_path))

        return measure_binary_size(binary_path)

    def measure_run_time(self, build: str = 'build', repeat=1, parallel=False) -> np.ndarray:
        binary_path = os.path.join(self.root_dir, build, self._file_name)
        if not os.path.isfile(binary_path):
            raise ValueError("Binary not found: {}".format(binary_path))

        run_times = []

        for workload in self._workloads:
            command = '{} {}'.format(binary_path, self._run_args.format(workload=workload))
            run_time = measure_run_time(command, repeat=repeat, parallel=False)
            run_times.append(np.mean(run_time))
        return np.array(run_times)

    def _get_metric(self, lines):
        for line in lines:
            m = re.match(self._metric_regex, line)
            if m:
                return float(m.group(1))
        return 0

    def measure_run_metric(self, build: str = 'build', repeat=1, parallel=False):
        binary_path = os.path.join(self.root_dir, build, self._file_name)
        if not os.path.isfile(binary_path):
            raise ValueError("Binary not found: {}".format(binary_path))

        run_times = []
        for workload in self._workloads:
            rts = []
            command = '{} {}'.format(binary_path, self._run_args.format(workload=workload))
            if parallel:
                procs = []
                for _ in range(repeat):
                    procs.append(Popen(command, stdout=PIPE, stderr=PIPE, shell=True))
                for proc in procs:
                    lines = proc.communicate()[0].decode('utf-8').split('\n')
                    rts.append(self._get_metric(lines))
            else:
                for _ in range(repeat):
                    proc = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
                    lines = proc.communicate()[0].decode('utf-8').split('\n')
                    rts.append(self._get_metric(lines))
            run_times.append(np.mean(rts))
        return np.array(run_times)

    def measure_run_score(self, build: str = 'build', repeat=None) -> np.ndarray:
        """
        Override this if you want an alternative run score.
        Otherwise, it returns the inverse of run times
        """
        if repeat is None:
            repeat = self._repeat_per_measure
        if self._metric_regex is None:
            return 1 / self.measure_run_time(build, repeat=repeat, parallel=self._parallel_measurement)
        else:
            return self.measure_run_metric(build, repeat=repeat, parallel=self._parallel_measurement)

    def normalize_run_score(self, run_times: np.ndarray, metric: str = 'O0-O3'):
        """
        :param run_times: raw run time
        :param metric:
            O0 - normalize based on O0, 0 when run_score is worst possible, 1 at O0;
            O3 - normalize based on O3, similar to O0;
            O0_O3 - normalize based on difference between O0 and O3
        :return:
            normalized run times
        """
        assert (self.O0_score is not None), "O0_score not initialized, please run calibrate_baseline()"
        assert (self.O3_score is not None), "O3_score not initialized, please run calibrate_baseline()"
        if metric == 'O0':
            return run_times / self.O0_score
        elif metric == 'O3':
            return run_times / self.O3_score
        elif metric == 'O0-O3':
            return (run_times - self.O0_score) / (self.O3_score - self.O0_score)
        else:
            # metric not recognized, return raw runtimes
            return run_times

    def measure_normalized_run_score(self, build: str = 'build', repeat=None, metric: str = 'O0-O3') -> np.ndarray:
        """A method combining measure_run_time and normalize_run_time, primarily for convenience"""
        if repeat is None:
            repeat = self._repeat_per_measure
        return self.normalize_run_score(self.measure_run_score(build, repeat=repeat), metric)
