import pickle

from matplotlib import pyplot as plt

from config.config import COREMARK_LOC
from benchmarks.coremark import CoreMarkProgram
import numpy as np

passes = ['-targetlibinfo', '-tbaa', '-scoped-noalias-aa', '-annotation2metadata', '-forceattrs', '-inferattrs',
           '-domtree', '-callsite-splitting', '-ipsccp', '-called-value-propagation', '-globalopt', '-domtree',
           '-mem2reg', '-deadargelim', '-domtree', '-basic-aa', '-aa', '-loops', '-instcombine', '-simplifycfg',
           '-globals-aa', '-inline', '-openmp-opt-cgscc', '-function-attrs', '-argpromotion', '-domtree', '-sroa',
           '-basic-aa', '-aa', '-memoryssa', '-early-cse-memssa', '-speculative-execution', '-aa', '-lazy-value-info',
           '-jump-threading', '-correlated-propagation', '-simplifycfg', '-domtree', '-aggressive-instcombine',
           '-basic-aa', '-aa', '-loops', '-instcombine', '-libcalls-shrinkwrap', '-loops', '-postdomtree',
           '-branch-prob', '-block-freq', '-pgo-memop-opt', '-basic-aa', '-aa', '-loops', '-tailcallelim',
           '-simplifycfg', '-reassociate', '-domtree', '-basic-aa', '-aa', '-memoryssa', '-loops', '-loop-simplify',
           '-lcssa', '-scalar-evolution', '-licm', '-loop-rotate', '-licm', '-simplifycfg', '-domtree', '-basic-aa',
           '-aa', '-loops', '-instcombine', '-loop-simplify', '-lcssa', '-scalar-evolution', '-loop-idiom', '-indvars',
           '-loop-deletion', '-loop-unroll', '-sroa', '-aa', '-mldst-motion', '-phi-values', '-aa', '-memdep', '-gvn',
           '-sccp', '-demanded-bits', '-bdce', '-basic-aa', '-aa', '-instcombine', '-lazy-value-info',
           '-jump-threading', '-correlated-propagation', '-postdomtree', '-adce', '-basic-aa', '-aa', '-memoryssa',
           '-memcpyopt', '-loops', '-dse', '-loop-simplify', '-lcssa', '-aa', '-scalar-evolution', '-licm',
           '-simplifycfg', '-domtree', '-basic-aa', '-aa', '-loops', '-instcombine', '-elim-avail-extern',
           '-rpo-function-attrs', '-globalopt', '-globaldce', '-globals-aa', '-domtree', '-float2int',
           '-lower-constant-intrinsics', '-loops', '-loop-simplify', '-lcssa', '-basic-aa', '-aa', '-scalar-evolution',
           '-loop-rotate', '-loop-distribute', '-postdomtree', '-branch-prob', '-block-freq', '-scalar-evolution',
           '-basic-aa', '-aa', '-demanded-bits', '-inject-tli-mappings', '-loop-vectorize', '-loop-simplify',
           '-scalar-evolution', '-aa', '-loop-load-elim', '-basic-aa', '-aa', '-instcombine', '-simplifycfg',
           '-domtree', '-loops', '-scalar-evolution', '-basic-aa', '-aa', '-demanded-bits']

prog = CoreMarkProgram(COREMARK_LOC)
prog.copy_build('clean_build', 'test_build')
prog.calibrate_baseline()
runtimes = []
for idx, p in enumerate(passes):
    prog.opt(p, 'test_build')
    prog.build_executable('test_build')
    run_time = np.mean(prog.measure_normalized_run_score('test_build'))
    runtimes.append(run_time)
    # print(f'Step {idx} - Pass {p} - {run_time}')
prog.remove_build('test_build')

with open('o3/o3', 'wb+') as f:
    pickle.dump(runtimes, f)

# plt.figure(figsize=(80, 4))
plt.plot(runtimes)
# plt.xticks(range(len(passes)), passes)

# for i in range(len(passes)):
#     plt.annotate(passes[i], (i, runtimes[i]))
plt.savefig('o3_agent.png')