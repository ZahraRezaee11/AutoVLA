import os, sys, json, glob, pickle, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'navsim'))
DATA = '/media/zara/6fbbe2fa-9fb0-49e6-bb18-67d7cd6c0e25/waymo_e2e'
sys.path.insert(0, DATA)  # for rater_feedback_utils
import yaml, torch, numpy as np
from transformers import AutoProcessor

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True)
p.add_argument('--config', default='config/training/waymo-sft-ours.yaml')
p.add_argument('--limit', type=int, default=0)  # 0 = all
args = p.parse_args()

config = yaml.safe_load(open(args.config))
mc = config['model']
processor = AutoProcessor.from_pretrained(mc['pretrained_model_path'], use_fast=True)

from dataset_utils.sft_dataset import SFTDataset
ds = SFTDataset({'json_dataset_path': './dataset/waymo_val_rated', 'sensor_data_path': None},
                mc, processor, using_cot=False)
print('eval samples:', len(ds))
ASTART = mc['tokens']['action_start_id']
atok = ds.action_tokenizer

# ---- model: same construction as training, then load ckpt ----
from models.autovla import SFTAutoVLA
model = SFTAutoVLA(config)
from peft import get_peft_model, LoraConfig, TaskType
lc = mc['lora']
model.autovla.vlm = get_peft_model(model.autovla.vlm, LoraConfig(modules_to_save=["embed_tokens", "lm_head"], 
    task_type=TaskType[lc.get('task_type', 'CAUSAL_LM')],
    target_modules=lc['target_modules'], r=lc['r'],
    lora_alpha=lc['alpha'], lora_dropout=lc['dropout'], bias=lc.get('bias', 'none')))
sd = torch.load(args.ckpt, map_location='cpu')['state_dict']
sd = { (k[len('model.'):] if k.startswith('model.') else k): v for k, v in sd.items() }
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f'load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected')
vlm = model.autovla.vlm.eval().to('cuda', dtype=torch.bfloat16)

# ---- rated ground truth ----
rated = {e['name']: e for e in pickle.load(open(f'{DATA}/rated_val.pkl', 'rb'))}

names, preds2hz, n_bad = [], [], 0
logprobs_all = []
items = list(range(len(ds)))
if args.limit: items = items[:args.limit]
with torch.no_grad():
    for i in items:
        s = ds[i]
        token_name = os.path.basename(s['data_path']).replace('.json', '') if 'data_path' in s else None
        prompt = (s['text'].split('<|im_start|>assistant')[0]
                  + '<|im_start|>assistant\n<answer>\nThe final output action is: ')
        inputs = processor(text=[prompt], videos=s['video_inputs'], padding=True,
                           return_tensors='pt').to('cuda')
        ALLOWED = list(range(ASTART, ASTART + atok.n_bins))
        K = int(os.environ.get('BESTOF_K', '8'))
        outs = [vlm.generate(**inputs, max_new_tokens=10, do_sample=True,
                             temperature=1.0, top_p=0.95, output_scores=True,
                             return_dict_in_generate=True,
                             prefix_allowed_tokens_fn=lambda b, ids: ALLOWED)
                for _ in range(K)]
        cand, lps = [], []
        for k in range(K):
            gen = outs[k].sequences[0, inputs['input_ids'].shape[1]:]
            lp = 0.0
            for t in range(min(len(gen), len(outs[k].scores))):
                lp += torch.log_softmax(outs[k].scores[t][0].float(), dim=-1)[gen[t]].item()
            lps.append(lp)
            act = [int(t) for t in gen][:10]
            if len(act) < 10 or any(t < ASTART for t in act):
                n_bad += 1
                act = [t for t in act if t >= ASTART]
                act = act + [act[-1] if act else ASTART] * (10 - len(act))
            traj = atok.decode_token_ids_to_trajectory(torch.tensor(act))
            cand.append(np.asarray(traj)[0, 1:, :2])
        names.append(token_name); preds2hz.append(np.stack(cand)); logprobs_all.append(lps)
        if (len(names)) % 25 == 0: print(f'{len(names)}/{len(items)}', flush=True)

preds2hz = np.stack(preds2hz)                                          # [B, K, 10, 2]
# ---- upsample 2Hz -> 4Hz (prepend origin, linear interp) ----
t2 = np.arange(0, 11) * 0.5                                            # 0..5.0
t4 = np.arange(1, 21) * 0.25                                           # 0.25..5.0
B, K = preds2hz.shape[:2]
with0 = np.concatenate([np.zeros((B, K, 1, 2)), preds2hz], axis=2)     # [B,K,11,2]
preds4hz = np.stack([np.stack([np.stack([np.interp(t4, t2, with0[b, k, :, d]) for d in range(2)], axis=1)
                     for k in range(K)]) for b in range(B)])            # [B,K,20,2]

# ---- RFS ----
import rater_feedback_utils as rfu
keep = [i for i, n in enumerate(names) if n in rated]
print(f'matched to rated: {len(keep)}/{len(names)} | short-output samples: {n_bad}')
inference = preds4hz[keep]
K = inference.shape[1]
probs = np.ones((len(keep), K)) / K
fr = [rated[names[i]] for i in keep]
past = np.stack([f['past'] for f in fr])
init_speed = np.linalg.norm((past[:, -1] - past[:, -2]) / 0.25, axis=1)
out = rfu.get_rater_feedback_score(
    inference_trajectories=inference, inference_probs=probs,
    rater_specified_trajectories=[f['raters'] for f in fr],
    rater_feedback_labels=[f['scores'] for f in fr],
    init_speed=init_speed)
rfs = np.asarray(out['rater_feedback_score'])
per_cand = []
for k in range(K):
    o = rfu.get_rater_feedback_score(
        inference_trajectories=inference[:, k:k+1], inference_probs=np.ones((len(keep), 1)),
        rater_specified_trajectories=[f['raters'] for f in fr],
        rater_feedback_labels=[f['scores'] for f in fr], init_speed=init_speed)
    per_cand.append(np.asarray(o['rater_feedback_score']))
per_cand = np.stack(per_cand, 1)                                        # [B, K]
lp_arr = np.asarray(logprobs_all)[keep]                                 # [B, K]
sel_lp = per_cand[np.arange(len(keep)), lp_arr.argmax(1)]
pw = np.linalg.norm(inference[:, :, None] - inference[:, None, :], axis=-1).mean(-1)  # [B,K,K]
sel_md = per_cand[np.arange(len(keep)), pw.sum(-1).argmin(1)]
np.savez('eval_bestofk_results.npz', names=np.array([names[i] for i in keep]), rfs=rfs,
         preds4hz=preds4hz[keep], logprobs=np.asarray(logprobs_all)[keep], per_cand=per_cand)
print(f'select-by-LOGPROB RFS: {sel_lp.mean():.3f}')
print(f'select-by-MEDOID  RFS: {sel_md.mean():.3f}')
print(f'best-of-{K} ORACLE RFS: {per_cand.max(1).mean():.3f}')
print(f'mean single-sample RFS: {per_cand.mean():.3f}')
print(f'worst-of-{K}: {per_cand.min(1).mean():.3f}')
print('\n=== MODEL RFS (478 rated frames) ===')
print(f'RFS mean: {rfs.mean():.3f}   (constant-velocity baseline: 7.027)')
print(f'min {rfs.min():.1f} | p25 {np.percentile(rfs,25):.1f} | median {np.median(rfs):.1f} | p75 {np.percentile(rfs,75):.1f} | max {rfs.max():.1f}')
print(f'fraction at floor (<=4): {(rfs<=4.0+1e-6).mean()*100:.1f}%  |  fraction >=8: {(rfs>=8).mean()*100:.1f}%')

