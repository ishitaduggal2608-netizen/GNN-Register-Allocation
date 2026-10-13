# Phase 6 model comparison

| Model | Test F1 (mean ± std, 3 seeds) |
|---|---:|
| GCN | 0.4886 ± 0.0763 |
| SAGE | 0.5285 ± 0.0852 |
| GAT | 0.1393 ± 0.1970 |

## Ablation of SAGE

| Feature subset | Test F1 (mean ± std, 3 seeds) |
|---|---:|
| all_features | 0.5285 ± 0.0852 |
| no_loop_depth | 0.3477 ± 0.2272 |
| no_move_related | 0.4197 ± 0.1751 |
| no_live_range_length | 0.3887 ± 0.2818 |
