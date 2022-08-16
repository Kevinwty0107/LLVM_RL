import os
import time
from subprocess import run, DEVNULL, Popen, STDOUT

from tempfile import TemporaryDirectory
from typing import List

import numpy as np


def measure_build_time(build_dir: str, clean_command: str, build_command_tpl: str, pass_list: List[str]):
    cwd = os.getcwd()
    os.chdir(build_dir)

    if pass_list:
        pass_file = os.path.abspath(os.path.join(build_dir, '_clang_pass_file.txt'))

        with open(pass_file, 'w') as fp:
            for pass_data in pass_list:
                fp.write(' '.join([str(a) for a in pass_data]) + '\n')
    else:
        pass_file = ''

    build_command = build_command_tpl.format(pass_file=pass_file)

    os.system(clean_command)

    start_time = time.perf_counter()
    os.system(build_command)
    time_taken = time.perf_counter() - start_time

    os.chdir(cwd)
    return time_taken


def measure_binary_size(file):
    stat = os.stat(file)
    return int(stat.st_size)


def measure_run_time(command, verbose=False, repeat=1, parallel=False):
    with TemporaryDirectory(prefix='llvm_rl_') as tempdir:
        cwd = os.getcwd()
        os.chdir(tempdir)
        times = []
        if verbose:
            stdout = STDOUT
        else:
            stdout = DEVNULL
        if parallel:
            procs = []
            for _ in range(repeat):
                procs.append(Popen(command, shell=True, stdout=stdout, stderr=stdout))
            for proc in procs:
                ru = os.wait4(proc.pid, 0)[2]
                times.append(ru.ru_utime)
        else:
            for _ in range(repeat):
                proc = Popen(command, shell=True, stdout=stdout, stderr=stdout)
                ru = os.wait4(proc.pid, 0)[2]
                times.append(ru.ru_utime)

        os.chdir(cwd)

    return np.mean(times)
