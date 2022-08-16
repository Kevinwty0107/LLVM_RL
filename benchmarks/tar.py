from llvm_rl.programs.program import Program


class TarProgram(Program):
    _binary_file = 'src/tar'
    _run_args = '-zcf out.tar.gz {workload}'
    _workloads = [
        'Nov09JnyExport.csv'
    ]
