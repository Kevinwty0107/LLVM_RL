import os
import unittest
from tempfile import TemporaryDirectory

from llvm_rl.programs.lepton import LeptonProgram


class TestLepton(unittest.TestCase):
    def setUp(self) -> None:
        src_dir = os.path.join(
            os.path.dirname(__file__),
            '..',
            '..',
            'programs',
            'lepton.bc'
        )

        build_dir = os.path.join(
            src_dir,
            'build'
        )

        self.program = LeptonProgram(
            src_dir=src_dir,
            build_dir=build_dir
        )

    def test_build(self):
        # self.skipTest("Skip build.")
        build_time = self.program.measure_build_time()
        print("Build time: {}".format(build_time))
        assert build_time > 10

    def test_binary_size(self):
        # self.skipTest("Skip binary size.")

        binary_size = self.program.measure_binary_size()
        print("Binary size: {}".format(binary_size))

        assert binary_size > 1024

    def test_run(self):
        # self.skipTest("Skip run.")

        with TemporaryDirectory(prefix='llvm_rl_test_') as tempdir:
            run_times = self.program.measure_run_time()

        print("Run times: {}".format(run_times))
        for run_time in run_times:
            assert run_time > 0.2


if __name__ == '__main__':
    unittest.main()
