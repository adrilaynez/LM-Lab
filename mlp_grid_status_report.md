# MLP Grid Training Status Report

**Overall Progress:** 92/108 Configurations Completed (85%)

## Currently Running / Missing Models (32 Total)

The grid is split across 4 parallel workers. Each worker has 8 configurations remaining.

### Worker 0 (Embedding 3)
- H32 (LR 0.1, 0.01) - *Running Now*
- H64 (LR 0.2, 0.1, 0.01)
- H128 (LR 0.2, 0.1, 0.01)

### Worker 1 (Embedding 6)
- H256 (LR 0.1, 0.01) - *Running Now*
- H512 (LR 0.2, 0.1, 0.01)
- H1024 (LR 0.2, 0.1, 0.01) - *Largest computation*

### Worker 2 (Embedding 16)
- H32 (LR 0.1, 0.01) - *Running Now*
- H64 (LR 0.2, 0.1, 0.01)
- H128 (LR 0.2, 0.1, 0.01)

### Worker 3 (Embedding 32)
- H256 (LR 0.1, 0.01) - *Running Now*
- H512 (LR 0.2, 0.1, 0.01)
- H1024 (LR 0.2, 0.1, 0.01) - *Largest computation*

## Performance & Health
- **Memory Check:** H1024 models for other embedding sizes (E2, E10) have successfully finished. This confirms GPU memory is sufficient.
- **Speed:** Each configuration takes ~20 minutes for 50,000 steps.
- **Estimate:** 8 rounds remaining x 20 mins ≈ **2 hours 40 minutes** to completion.

Status: **HEALTHY / RUNNING**
