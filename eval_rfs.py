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
        out = vlm.generate(**inputs, max_new_tokens=10, do_sample=False,
                           prefix_allowed_tokens_fn=lambda b, ids: ALLOWED)
        gen = out[0, inputs['input_ids'].shape[1]:]
        act = [int(t) for t in gen][:10]
        if len(act) < 10 or any(t < ASTART for t in act):
            n_bad += 1
            act = [t for t in act if t >= ASTART]
            act = act + [act[-1] if act else ASTART] * (10 - len(act))
        traj = atok.decode_token_ids_to_trajectory(torch.tensor(act))  # [1, 11, 3] incl origin
        traj = np.asarray(traj)[0, 1:, :2]                             # [10, 2] @2Hz
        names.append(token_name); preds2hz.append(traj)
        if (len(names)) % 25 == 0: print(f'{len(names)}/{len(items)}', flush=True)

preds2hz = np.stack(preds2hz)                                          # [B, 10, 2]
# ---- upsample 2Hz -> 4Hz (prepend origin, linear interp) ----
t2 = np.arange(0, 11) * 0.5                                            # 0..5.0
t4 = np.arange(1, 21) * 0.25                                           # 0.25..5.0
with0 = np.concatenate([np.zeros((len(preds2hz), 1, 2)), preds2hz], axis=1)  # [B,11,2]
preds4hz = np.stack([np.stack([np.interp(t4, t2, w[:, d]) for d in range(2)], axis=1)
                     for w in with0])                                  # [B,20,2]

# ---- RFS ----
import rater_feedback_utils as rfu
keep = [i for i, n in enumerate(names) if n in rated]
print(f'matched to rated: {len(keep)}/{len(names)} | short-output samples: {n_bad}')
inference = preds4hz[keep][:, None, :, :]
probs = np.ones((len(keep), 1))
fr = [rated[names[i]] for i in keep]
past = np.stack([f['past'] for f in fr])
init_speed = np.linalg.norm((past[:, -1] - past[:, -2]) / 0.25, axis=1)
out = rfu.get_rater_feedback_score(
    inference_trajectories=inference, inference_probs=probs,
    rater_specified_trajectories=[f['raters'] for f in fr],
    rater_feedback_labels=[f['scores'] for f in fr],
    init_speed=init_speed)
rfs = np.asarray(out['rater_feedback_score'])
np.savez('eval_rfs_results.npz', names=np.array([names[i] for i in keep]), rfs=rfs, preds4hz=preds4hz[keep])
print('\n=== MODEL RFS (478 rated frames) ===')
print(f'RFS mean: {rfs.mean():.3f}   (constant-velocity baseline: 7.027)')
print(f'min {rfs.min():.1f} | p25 {np.percentile(rfs,25):.1f} | median {np.median(rfs):.1f} | p75 {np.percentile(rfs,75):.1f} | max {rfs.max():.1f}')
print(f'fraction at floor (<=4): {(rfs<=4.0+1e-6).mean()*100:.1f}%  |  fraction >=8: {(rfs>=8).mean()*100:.1f}%')

