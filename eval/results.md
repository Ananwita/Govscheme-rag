Embedder: `sentence-transformers:intfloat/multilingual-e5-small`  |  questions: 44 (22 EN, 22 HI)

| Configuration | Subset | Hit@1 | Hit@3 | MRR | Section@1 | FactRecall |
|---|---|---|---|---|---|---|
| hybrid | all | 0.955 | 1.0 | 0.977 | 0.864 | 0.909 |
| hybrid | en | 1.0 | 1.0 | 1.0 | 0.955 | 0.955 |
| hybrid | hi | 0.909 | 1.0 | 0.955 | 0.773 | 0.864 |
| dense_only | all | 0.955 | 0.977 | 0.97 | 0.841 | - |
| dense_only | en | 0.955 | 1.0 | 0.977 | 0.864 | - |
| dense_only | hi | 0.955 | 0.955 | 0.964 | 0.818 | - |
| bm25_only | all | 0.864 | 1.0 | 0.924 | 0.75 | - |
| bm25_only | en | 0.909 | 1.0 | 0.955 | 0.818 | - |
| bm25_only | hi | 0.818 | 1.0 | 0.894 | 0.682 | - |
| cross_lingual | all | 0.864 | 0.955 | 0.919 | 0.727 | - |
| cross_lingual | en | 0.955 | 1.0 | 0.977 | 0.818 | - |
| cross_lingual | hi | 0.773 | 0.909 | 0.861 | 0.636 | - |

### Hybrid misses at rank 1
- `hi05` expected **ayushman_bharat_pmjay** got rank 2 (top1: pmjjby_pmsby/पीएमएसबीवाई की पात्रता)
- `hi18` expected **pm_vishwakarma** got rank 2 (top1: pm_kisan/अपात्रता (कौन पात्र नहीं है))
