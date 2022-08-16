from config.passes import ALL_PASSES
from config.config import VALID_PASS_LIST_LOC
from benchmarks.coremark import CoreMarkProgram, COREMARK_LOC


def sanitize(program, pass_list, time_cap=10):
    _, _, exist_clean = program.check_build('clean_build')
    if not exist_clean:
        program.build_executable('clean_build')
    program.copy_build(target_build='temp_build')

    # Passes that can produce correct bc and executable
    valid = []
    # Passes that takes more than time_cap to run
    costly = []
    # Passes that fails to build bc
    failed = []
    # Passes that produce bc, but gives error when building executable
    danger = []

    for p in pass_list:
        build_time = program.opt(f'-{p}', 'temp_build', time_cap=time_cap)
        # file_to_check = os.path.join(program.root_dir, 'build', f'{program._file_name}1.bc')
        if build_time == -1:
            costly.append(p)
        elif build_time == -2:
            failed.append(p)
        else:
            # print(build_time)
            ret_code = program.build_executable('temp_build')
            if ret_code != 0:
                danger.append(p)
                program.copy_build(target_build='temp_build')
            else:
                valid.append(p)
    with open(VALID_PASS_LIST_LOC, 'w+') as f:
        f.write(f'FAILED_PASSES = {failed}\n\n')
        f.write(f'COSTLY_PASSES = {costly}\n\n')
        f.write(f'VALID_PASSES = {valid}\n\n')
        f.write(f'DANGEROUS_PASSES = {danger}\n\n')

if __name__ == '__main__':
    program = CoreMarkProgram(COREMARK_LOC)
    sanitize(program, ALL_PASSES)