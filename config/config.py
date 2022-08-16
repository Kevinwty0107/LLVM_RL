# LLVM_RL_LOC = '/mnt/d/Cambridge/DissProjectM/llvm_rl'
LLVM_RL_LOC = '/mnt/c/yilin/llvm_rl'

# Environment Related
COUNTED_INST = [
    'Add',
    # 'Sub',
    'Mul',
    # 'Div',
    # 'And',
    'Alloca',
    # 'Unreachable',
    'Br',
    'Call',
    # 'Ret',
    'Load',
    'Store',
    'ICmp',
    # 'basic',
    # 'non-external'
    # 'instructions'
]

VOCAB_LOC = f'{LLVM_RL_LOC}/vocab/seedEmbeddingVocab-300-llvm12.txt'

# Benchmark Related
BENCHMARK_LOC = f'{LLVM_RL_LOC}/benchmarks'

LEPTON_LOC = f'{BENCHMARK_LOC}/lepton'

STANFORD_LOC = f'{BENCHMARK_LOC}/stanford'
STANFORD_PROG = ['Bubblesort', 'FloatMM', 'IntMM', 'RealMM', 'Oscar', 'Perm', 'Queens', 'Towers', 'Treesort', 'Puzzle']

COREMARK_LOC = f'{BENCHMARK_LOC}/ansibench/coremark'

HIMENO_LOC = f'{BENCHMARK_LOC}/himeno'

POLYBENCH_LOC = f'{BENCHMARK_LOC}/polybench'
POLYBENCH_PROGS = [
    'gemm',  # 2.946242
    'gemver',  # 0.041802
    'gesummv',  # 0.007068
    'symm',  # 3.641413
    'syr2k',  # 4.714283
    'syrk',  # 1.843448
    'trmm',  # 1.908813
    '2mm',  # 4.796188
    '3mm',  # 7.430869 [Costly]
    'atax',  # 0.016489
    'bicg',  # 0.0155
    'doitgen',  # 1.180033
    'mvt',  # 0.029928
    # 'cholesky',
    'durbin',  # 0.009014
    # 'gramschmidt', # 5.824078
    'lu',  # 9.466454 [Very Costly]
    'ludcmp',  # 10.76426 [Very Costly]
    'trisolv',  # 0.008076
    # 'correlation', # 4.570999
    'covariance',  # 5.398919
    # 'deriche', # 0.344755
    'floyd-warshall',  # 72.906361 [Very Very Costly]
    'nussinov',  # 9.241275 [Very Costly]
    'adi',  # 10.534314 [Very Costly]
    'fdtd-2d',  # 5.048663
    'heat-3d',  # 12.752966 [Very Costly]
    'jacobi-1d',  # 0.003984
    'jacobi-2d',  # 6.312519 [Costly]
    'seidel-2d'  # 20.876238 [Very Very Costly]
]

POLYBENCH_PROGS_BY_TIME = [
    [
        'jacobi-1d',  # 0.003984
        'gesummv',  # 0.007068
        'trisolv',  # 0.008076
        'durbin',  # 0.009014
        'bicg',  # 0.0155
        'atax',  # 0.016489
        'mvt',  # 0.029928
        'gemver',  # 0.041802
        'deriche', # 0.344755
    ],
    [
        'doitgen',  # 1.180033
        'syrk',  # 1.843448
        'trmm',  # 1.908813
        'gemm',  # 2.946242
        'symm',  # 3.641413
    ],
    [
        'correlation', # 4.570999
        'syr2k',  # 4.714283
        '2mm',  # 4.796188
        'fdtd-2d',  # 5.048663
        'covariance',  # 5.398919
        'gramschmidt', # 5.824078
        'jacobi-2d',  # 6.312519 [Costly]
        '3mm',  # 7.430869 [Costly]
    ],
    [
        'nussinov',  # 9.241275 [Very Costly]
        'lu',  # 9.466454 [Very Costly]
        'adi',  # 10.534314 [Very Costly]
        'ludcmp',  # 10.76426 [Very Costly]
        'heat-3d',  # 12.752966 [Very Costly]
        'seidel-2d'  # 20.876238 [Very Very Costly]
        'cholesky', # [Very Costly]
        'floyd-warshall',  # 72.906361 [Very Very Costly]
    ]
]

POLYBENCH_LM = ['deriche', 'gramschmidt', 'correlation', 'cholesky']

# Agent Related
STATE_DICT_LOC = f'{LLVM_RL_LOC}/agents/state_dict'
DQN_STATE_DICT_LOC = f'{STATE_DICT_LOC}/dqn_q'
DQN_TARGET_STATE_DICT_LOC = f'{STATE_DICT_LOC}/dqn_target'

# Pass Related
PASS_LIST_LOC = f'{LLVM_RL_LOC}/config/passes.py'
VALID_PASS_LIST_LOC = f'{LLVM_RL_LOC}/config/valid_passes.py'
O3_PASS_LIST_LOC = f'{LLVM_RL_LOC}/config/O3_passes.py'
O3_PASS_GROUP_LOC = f'{LLVM_RL_LOC}/config/O3_pass_groups.py'
