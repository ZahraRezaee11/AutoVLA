# WOD-E2E on a single RTX 5090: AutoVLA reproduction + RFS-reward RL

Zara's working branch on top of [AutoVLA](https://github.com/ucla-mobility/AutoVLA) for the
[Waymo Vision-based End-to-End Driving Challenge](https://waymo.com/open/challenges/2025/e2e-driving/).
Everything below runs on one RTX 5090 (32 GB), Waymo data only, no CoT annotations.

## Results (478 rated validation frames, local RFS metric)

| Method | RFS | at floor (<=4) |
|---|---|---|
| Constant velocity baseline | 7.027 | 26.9% |
| SFT greedy (LoRA + trained embeddings, 1 epoch on training split) | 7.013 | 26.8% |
| + GRPO with pseudo-RFS reward, v1 (G=8) | 7.120 | 23.4% |
| + GRPO v2 (G=16, top-adv backward, cosine lr, scene skip-list) | 7.145 | 23.0% |
| + GRPO v3 (longer, T=1.1) | 7.149 | 22.6% |
| **medoid-of-8 selection at inference** | **7.386** | - |
| best-of-8 oracle (selection ceiling) | 8.444 | - |
| GT-oracle (imitation ceiling, measured) | 8.175 | 9.4% |
| Best-rater-per-frame (preference ceiling, measured) | 9.600 | - |
| Official references: NaiveEMMA 7.53, AutoVLA paper 7.556, leaderboard top ~8.17 | | |

## What is ours

- `ours_scripts/build_*_lmdb.py`: OOM-safe LMDB builders (upstream builder needs >30 GB RAM; ours streams at ~65 rec/s, 427 MB output)
- `make_eval_samples.py`: builds the 478-frame rated eval set (the default sampler grid misses every rated frame)
- `ours_scripts/local_rfs.py` + `rater_feedback_utils.py`: local Rater Feedback Score, since the server allows 6 submissions/month
- `tools/run_sft_lora.py`: LoRA SFT with `modules_to_save=[embed_tokens, lm_head]`. Key bug found: LoRA-only SFT leaves the 2048 new action-token embeddings frozen at random init, which collapses the model to an input-independent marginal (RFS 4.1). Training the embeddings fixes it (val_loss 3.30 -> 1.14, RFS 4.1 -> 7.0)
- `ours_scripts/gt_rfs_reward.py`: pseudo-RFS reward on the training split (GT trajectory as a single rater with score 10, official trust-region geometry)
- `tools/run_rft_waymo.py`: single-GPU GRPO. Per-prompt groups (upstream normalizes across GPUs, which yields zero advantage on one GPU), G=16 rollouts with backward on top-8 |advantage|, cosine lr, degenerate-scene skip-list
- `eval_rfs.py`, `eval_rfs_bestofk.py`: constrained-decode evaluation (generation restricted to action-token range), best-of-K oracle and selectors (logprob, medoid)
- `make_selector_data.py`: builds training data for a learned candidate selector (phase in progress; selection ceiling is 8.44)

## Pipeline

1. Extract images + build LMDB (`ours_scripts/`), generate samples (`tools/preprocessing/nocot_sample_generation.py` with `config/dataset/waymo-*-ours.yaml`)
2. SFT: `python tools/run_sft_lora.py --config training/waymo-sft-train`
3. RL: `python tools/run_rft_waymo.py --config training/waymo-rft-v3`
4. Eval: `python eval_rfs.py --ckpt <ckpt> --config config/training/waymo-rft-v3.yaml`

Checkpoints and datasets are not in the repo (size); ask me for the weights.

## Pipeline

1. Extract images + build LMDB (`ours_scripts/`), generate samples (`tools/preprocessing/nocot_sample_generation.py` with `config/dataset/waymo-*-ours.yaml`)
2. SFT: `python tools/run_sft_lora.py --config training/waymo-sft-train`
3. RL: `python tools/run_rft_waymo.py --config training/waymo-rft-v3`
4. Eval: `python eval_rfs.py --ckpt <ckpt> --config config/training/waymo-rft-v3.yaml`

Checkpoints and datasets are not in the repo (size); ask me for the weights.
