from config.O3_passes import O3_PASS_SEQUENCE
from config.config import O3_PASS_GROUP_LOC


def split(pass_list=O3_PASS_SEQUENCE, delimiter='simplifycfg'):
    pass_groups = []
    pass_group = []
    for p in pass_list:
        pass_group.append(p)
        if p == delimiter:
            pass_groups.append(pass_group)
            pass_group = []
    return pass_groups


def pass_group_to_pass(pass_group):
    return ' -'.join(pass_group)


if __name__ == '__main__':
    pass_groups = split()
    with open(O3_PASS_GROUP_LOC, 'w+') as f:
        f.write(f'O3_PASS_GROUPS = {pass_groups}\n\n')
        f.write(f'O3_PASS_GROUPS_AS_PASSES = {[pass_group_to_pass(p) for p in pass_groups]}\n\n')
