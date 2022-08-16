from programs.program import Program
from config.config import HIMENO_LOC


class HimenoProgram(Program):
    _file_name = 'himenobmtxpa'
    _run_args = 'S'
    _metric_regex = r'^\s*Score based on Pentium III 600MHz using Fortran 77: ([0-9]+(.[0-9]+)?)'
