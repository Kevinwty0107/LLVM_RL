O3_PASSES = ['aa', 'adce', 'aggressive-instcombine', 'alignment-from-assumptions', 'annotation-remarks',
             'annotation2metadata', 'argpromotion', 'assumption-cache-tracker', 'barrier', 'basic-aa', 'basiccg',
             'bdce', 'block-freq', 'branch-prob', 'called-value-propagation', 'callsite-splitting', 'cg-profile',
             'constmerge', 'correlated-propagation', 'deadargelim', 'demanded-bits', 'div-rem-pairs', 'domtree', 'dse',
             'early-cse', 'early-cse-memssa', 'elim-avail-extern', 'float2int', 'forceattrs', 'function-attrs',
             'globaldce', 'globalopt', 'globals-aa', 'gvn', 'indvars', 'inferattrs', 'inject-tli-mappings', 'inline',
             'instcombine', 'instsimplify', 'ipsccp', 'jump-threading', 'lazy-block-freq', 'lazy-branch-prob',
             'lazy-value-info', 'lcssa', 'lcssa-verification', 'libcalls-shrinkwrap', 'licm', 'loop-accesses',
             'loop-deletion', 'loop-distribute', 'loop-idiom', 'loop-load-elim', 'loop-rotate', 'loop-simplify',
             'loop-sink', 'loop-unroll', 'loop-unswitch', 'loop-vectorize', 'loops', 'lower-constant-intrinsics',
             'lower-expect', 'mem2reg', 'memcpyopt', 'memdep', 'memoryssa', 'mldst-motion', 'openmp-opt-cgscc',
             'opt-remark-emitter', 'pgo-memop-opt', 'phi-values', 'postdomtree', 'profile-summary-info', 'prune-eh',
             'reassociate', 'rpo-function-attrs', 'scalar-evolution', 'sccp', 'scoped-noalias-aa', 'simplifycfg',
             'slp-vectorizer', 'speculative-execution', 'sroa', 'strip-dead-prototypes', 'tailcallelim',
             'targetlibinfo', 'targetpassconfig', 'tbaa', 'transform-warning', 'tti', 'vector-combine', 'verify']

VALID_O3_PASSES = ['targetlibinfo', 'globals-aa', 'scalar-evolution', 'memoryssa', 'scoped-noalias-aa', 'demanded-bits',
                   'branch-prob', 'domtree', 'lazy-value-info', 'postdomtree', 'basic-aa', 'loops', 'aa', 'block-freq',
                   'tbaa', 'phi-values', 'memdep', 'globalopt', 'loop-rotate', 'forceattrs', 'reassociate',
                   'libcalls-shrinkwrap', 'div-rem-pairs', 'early-cse-memssa', 'sroa', 'memcpyopt', 'jump-threading',
                   'openmp-opt-cgscc', 'loop-distribute', 'adce', 'function-attrs', 'indvars', 'float2int',
                   'called-value-propagation', 'loop-unroll', 'pgo-memop-opt', 'correlated-propagation', 'lcssa',
                   'tailcallelim', 'sccp', 'aggressive-instcombine', 'instcombine', 'loop-load-elim', 'instsimplify',
                   'strip-dead-prototypes', 'argpromotion', 'vector-combine', 'loop-idiom', 'loop-vectorize',
                   'constmerge', 'mem2reg', 'globaldce', 'simplifycfg', 'speculative-execution',
                   'lower-constant-intrinsics', 'gvn', 'inferattrs', 'bdce', 'loop-deletion', 'licm',
                   'elim-avail-extern', 'early-cse', 'inject-tli-mappings', 'lower-expect', 'inline', 'deadargelim',
                   'alignment-from-assumptions', 'loop-simplify', 'ipsccp', 'slp-vectorizer', 'loop-sink',
                   'mldst-motion', 'callsite-splitting', 'dse']

IRRELEVANT_PASSES = ['annotation2metadata', 'warning', 'verify', 'annotation-remarks', 'rpo-function-attrs',
                     'cg-profile']

O3_PASS_SEQUENCE = ['targetlibinfo', 'tbaa', 'scoped-noalias-aa', 'annotation2metadata', 'forceattrs', 'inferattrs',
                    'domtree', 'callsite-splitting', 'ipsccp', 'called-value-propagation', 'globalopt', 'domtree',
                    'mem2reg', 'deadargelim', 'domtree', 'basic-aa', 'aa', 'loops', 'instcombine', 'simplifycfg',
                    'globals-aa', 'inline', 'openmp-opt-cgscc', 'function-attrs', 'argpromotion', 'domtree', 'sroa',
                    'basic-aa', 'aa', 'memoryssa', 'early-cse-memssa', 'speculative-execution', 'aa', 'lazy-value-info',
                    'jump-threading', 'correlated-propagation', 'simplifycfg', 'domtree', 'aggressive-instcombine',
                    'basic-aa', 'aa', 'loops', 'instcombine', 'libcalls-shrinkwrap', 'loops', 'postdomtree',
                    'branch-prob', 'block-freq', 'pgo-memop-opt', 'basic-aa', 'aa', 'loops', 'tailcallelim',
                    'simplifycfg', 'reassociate', 'domtree', 'basic-aa', 'aa', 'memoryssa', 'loops', 'loop-simplify',
                    'lcssa', 'scalar-evolution', 'licm', 'loop-rotate', 'licm', 'simplifycfg', 'domtree', 'basic-aa',
                    'aa', 'loops', 'instcombine', 'loop-simplify', 'lcssa', 'scalar-evolution', 'loop-idiom', 'indvars',
                    'loop-deletion', 'loop-unroll', 'sroa', 'aa', 'mldst-motion', 'phi-values', 'aa', 'memdep', 'gvn',
                    'sccp', 'demanded-bits', 'bdce', 'basic-aa', 'aa', 'instcombine', 'lazy-value-info',
                    'jump-threading', 'correlated-propagation', 'postdomtree', 'adce', 'basic-aa', 'aa', 'memoryssa',
                    'memcpyopt', 'loops', 'dse', 'loop-simplify', 'lcssa', 'aa', 'scalar-evolution', 'licm',
                    'simplifycfg', 'domtree', 'basic-aa', 'aa', 'loops', 'instcombine', 'elim-avail-extern',
                    'rpo-function-attrs', 'globalopt', 'globaldce', 'globals-aa', 'domtree', 'float2int',
                    'lower-constant-intrinsics', 'loops', 'loop-simplify', 'lcssa', 'basic-aa', 'aa',
                    'scalar-evolution', 'loop-rotate', 'loop-distribute', 'postdomtree', 'branch-prob', 'block-freq',
                    'scalar-evolution', 'basic-aa', 'aa', 'demanded-bits', 'inject-tli-mappings', 'loop-vectorize',
                    'loop-simplify', 'scalar-evolution', 'aa', 'loop-load-elim', 'basic-aa', 'aa', 'instcombine',
                    'simplifycfg', 'domtree', 'loops', 'scalar-evolution', 'basic-aa', 'aa', 'demanded-bits']
