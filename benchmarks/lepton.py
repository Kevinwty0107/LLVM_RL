import os
import re

from programs.project import Project
from config.config import LEPTON_LOC


class LeptonProject(Project):
    _file_name = 'lepton'
    _run_args = '{workload} out.lep'
    _workloads = [
        'lepton/android.jpg',
        'lepton/androidcrop.jpg',
        'lepton/androidcropoptions.jpg',
        'lepton/androidprogressive.jpg',
        'lepton/androidtrail.jpg',
        'lepton/arithmetic.jpg',
        'lepton/badzerorun.jpg',
        'lepton/colorswap.jpg',
        'lepton/gray2sf.jpg',
        'lepton/grayscale.jpg',
        # 'lepton/hq.jpg',
        'lepton/iphone.jpg',
        'lepton/iphonecity.jpg',
        'lepton/iphonecrop.jpg',
        'lepton/iphonecrop2.jpg',
        'lepton/iphoneprogressive.jpg',
        'lepton/iphoneprogressive2.jpg',
        'lepton/narrowrst.jpg',
        'lepton/nofsync.jpg',
        # 'lepton/slrcity.jpg',
        'lepton/slrhills.jpg',
        'lepton/slrindoor.jpg',
        'lepton/trailingrst.jpg',
        'lepton/trailingrst2.jpg',
        # 'lepton/trunc.jpg',
        # 'lepton/truncatedzerorun.jpg',
    ]


def setup_lepton():
    cwd = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    if os.path.exists('lepton'):
        os.chdir('lepton')
        os.system('git reset --hard origin/master')
    else:
        os.system('git clone git@github.com:dropbox/lepton.git')
        os.chdir('lepton')
    lines = []
    with open('CMakeLists.txt', 'r') as f:
        for line in f:
            line = re.sub(r'set\(CMAKE_C_FLAGS [\'"](.*)[\'"]\)',
                          r'set(CMAKE_C_FLAGS "\1 ${OptLvl} -flto")',
                          line, flags=re.IGNORECASE)
            line = re.sub(r'set\(CMAKE_CXX_FLAGS [\'"](.*)[\'"]\)',
                          r'set(CMAKE_CXX_FLAGS "\1 ${OptLvl} -flto")',
                          line, flags=re.IGNORECASE)
            line = re.sub('project\(lepton\)',
                          'project(lepton)\n'
                          'set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} "-flto '
                          '-fuse-ld=gold -Wl,-plugin-opt=emit-llvm")',
                          line, flags=re.IGNORECASE)
            lines.append(line)
    with open('CMakeLists.txt', 'w') as f:
        f.writelines(lines)
    project = LeptonProject(os.getcwd(), 'build-no-opt')
    os.chdir(cwd)
    return project

def setup_existing_lepton():
    return LeptonProject(LEPTON_LOC)