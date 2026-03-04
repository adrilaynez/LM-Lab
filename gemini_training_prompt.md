# Gemini Flash Training Task: Complete Pedagogical MLP Training

## Context
You are tasked with completing the training of pedagogical MLP models for a language modeling research project. The training includes Group 2 (Stability Technique Grid) AND Group 3 (Big Models).

## Current Status
- **Group 2**: 9/20 models trained successfully (11 remaining)
- **Group 3**: 0 models trained (all need to be trained)
- **GPU Constraint**: Limited GPU memory - train sequentially, not in parallel
- **Locations**: 
  - Group 2: `c:\Projects\LM-Lab\checkpoints\pedagogical\stability_grid\`
  - Group 3: `c:\Projects\LM-Lab\checkpoints\pedagogical\big_models\`

## Models Already Complete ✅
- L1_none.pt, L1_kaiming.pt, L1_kaiming+BN.pt, L1_kaiming+BN+residual.pt
- L2_none.pt, L2_kaiming.pt, L2_kaiming+BN.pt, L2_kaiming+BN+residual.pt  
- L3_none.pt

## Models to Train 🔄

### Group 2 - Stability Grid (11 remaining models)
```
L3_kaiming, L3_kaiming+BN, L3_kaiming+BN+residual
L4_none, L4_kaiming, L4_kaiming+BN, L4_kaiming+BN+residual
L6_none, L6_kaiming, L6_kaiming+BN, L6_kaiming+BN+residual
```

### Group 3 - Big Models (ALL models need training)
Group 3 contains large models with:
- Hidden sizes: 256, 512
- Layers: 4, 6  
- Context sizes: 4, 8, 16, 32, 64, 128, 256
- Embedding dims: scaled with context (max(16, context//2))
- All use: Kaiming init + BatchNorm + Residual
- Training steps: 100,000 (vs 80,000 for Groups 1&2)

Expected ~20-30 big models (filtered to <25M parameters each)

## Instructions

### 1. Complete Group 2 First
Run the existing training script for Group 2:
```bash
cd c:\Projects\LM-Lab
python train_pedagogical.py --group 2
```

### 2. Then Train Group 3 (Big Models)
After Group 2 completes, run Group 3:
```bash
cd c:\Projects\LM-Lab
python train_pedagogical.py --group 3
```

### 3. Monitor Progress Efficiently
- Check progress every 10-15 minutes (big models take longer)
- Use these commands to check for new models:
```powershell
# Group 2 progress
Get-ChildItem "c:\Projects\LM-Lab\checkpoints\pedagogical\stability_grid" | Sort-Object LastWriteTime -Descending | Select-Object -First 3

# Group 3 progress  
Get-ChildItem "c:\Projects\LM-Lab\checkpoints\pedagogical\big_models" | Sort-Object LastWriteTime -Descending | Select-Object -First 3
```

### 4. Handle Potential Issues
- If training gets stuck (no progress for 20+ minutes for big models), kill the process and restart
- Use: `taskkill /F /PID <process_id>`
- Then restart with appropriate group: `python train_pedagogical.py --group 2` or `--group 3`

### 5. Expected Timeline
- **Group 2**: 11 models × 5-10 minutes = ~1-2 hours
- **Group 3**: ~20-30 big models × 10-20 minutes = ~4-8 hours  
- **Total**: ~5-10 hours of training time
- Group 2 should complete by ~20:00 (8 PM)
- Group 3 may run overnight into early morning

### 6. Final Verification
When all models are complete, verify both groups:
```powershell
# Group 2: Should show 20 .pt files
Get-ChildItem "c:\Projects\LM-Lab\checkpoints\pedagogical\stability_grid\*.pt" | Measure-Object

# Group 3: Should show 20-30 .pt files  
Get-ChildItem "c:\Projects\LM-Lab\checkpoints\pedagogical\big_models\*.pt" | Measure-Object
```

### 7. Success Criteria
**Group 2 (Stability Grid):**
- All 20 model files exist in stability_grid directory
- Each file should be 50KB-200KB in size

**Group 3 (Big Models):**
- All ~20-30 big model files exist in big_models directory  
- Files will be larger (500KB-5MB) due to bigger architectures
- All models use Kaiming+BN+Residual configuration

## Important Notes
- **DO NOT** try parallel training - GPU memory is limited
- **DO NOT** modify the training script - use the existing working one
- **DO** restart training if it appears stuck (especially for big models)
- **DO** monitor progress efficiently (big models take much longer)
- **DO** complete Group 2 first before starting Group 3

## Expected Final State

### Group 2 (stability_grid/):
```
L1_none.pt, L1_kaiming.pt, L1_kaiming+BN.pt, L1_kaiming+BN+residual.pt
L2_none.pt, L2_kaiming.pt, L2_kaiming+BN.pt, L2_kaiming+BN+residual.pt
L3_none.pt, L3_kaiming.pt, L3_kaiming+BN.pt, L3_kaiming+BN+residual.pt
L4_none.pt, L4_kaiming.pt, L4_kaiming+BN.pt, L4_kaiming+BN+residual.pt
L6_none.pt, L6_kaiming.pt, L6_kaiming+BN.pt, L6_kaiming+BN+residual.pt
```

### Group 3 (big_models/):
```
big_H256_L4_CTX4_E16.pt, big_H256_L4_CTX8_E16.pt, big_H256_L4_CTX16_E16.pt, ...
big_H256_L6_CTX4_E16.pt, big_H256_L6_CTX8_E16.pt, big_H256_L6_CTX16_E16.pt, ...
big_H512_L4_CTX4_E16.pt, big_H512_L4_CTX8_E16.pt, big_H512_L4_CTX16_E16.pt, ...
big_H512_L6_CTX4_E16.pt, big_H512_L6_CTX8_E16.pt, big_H512_L6_CTX16_E16.pt, ...
(All context sizes: 4, 8, 16, 32, 64, 128, 256)
```

## Troubleshooting
- If script fails with "too many indices" error, the batch handling needs fixing
- If script fails with memory error, reduce batch size in the script
- If process hangs, kill and restart

Your goal is to ensure ALL models are successfully trained and saved:
- **Group 2**: 20 stability grid models  
- **Group 3**: ~20-30 big models

Report back when complete or if you encounter any issues. This is a substantial training task that will likely run overnight.
