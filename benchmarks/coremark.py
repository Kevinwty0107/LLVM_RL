from programs.program import Program


class CoreMarkProgram(Program):
    _file_name = 'coremark'
    _build_bc_command = 'clang {opt_flags} -flto -fuse-ld=ld -Wl,-plugin-opt=emit-llvm ' \
                        '../src/* -I../include -DITERATIONS=10000 -o {file_name}.bc '
    _metric_regex = r"^Iterations\/Sec\s*:\s*([0-9]*(?:\.[0-9]*)?)"
    _repeat_per_measure = 1
