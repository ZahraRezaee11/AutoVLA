import os, sys, argparse
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./navsim'))
DATA = '/media/zara/6fbbe2fa-9fb0-49e6-bb18-67d7cd6c0e25/waymo_e2e'
sys.path.insert(0, DATA)

import yaml, torch, numpy as np
from transformers import AutoProcessor
from peft import get_peft_model, LoraConfig, TaskType
from dataset_utils.sft_dataset import SFTDataset
from models.autovla import SFTAutoVLA
from gt_rfs_reward import gt_rfs_reward

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True)
p.add_argument('--config', default='config/training/waymo-rft-v3.yaml')
p.add_argument('--limit', type=int, default=6000)
p.add_argument('--K', type=int, default=8)
p.add_argument('--out', default='selector_data.npz')
args = p.parse_args()

config = yaml.safe_load(open(args.config))
mc = config['model']
processor = AutoProcessor.from_pretrained(mc['pretrained_model_path'], use_fast=True)
ds = SFTDataset({'json_dataset_path': './dataset/waymo_train', 'sensor_data_path': None},
                mc, processor, using_cot=False)
print('train samples:', len(ds))
ASTART = mc['tokens']['action_start_id']
atok = ds.action_tokenizer

model = SFTAutoVLA(config)
lc = mc['lora']
model.autovla.vlm = get_peft_model(model.autovla.vlm, LoraConfig(
    modules_to_save=["embed_tokens", "lm_head"],
    task_type=TaskType[lc.get('task_type', 'CAUSAL_LM')],
    target_modules=lc['target_modules'], r=lc['r'],
    lora_alpha=lc['alpha'], lora_dropout=lc['dropout'], bias=lc.get('bias', 'none')))
sd = torch.load(args.ckpt, map_location='cpu')['state_dict']
sd = { (k[len('model.'):] if k.startswith('model.') else k): v for k, v in sd.items() }
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f'load: {len(missing)} missing, {len(unexpected)} unexpected')
vlm = model.autovla.vlm.eval().to('cuda', dtype=torch.bfloat16)

T2 = np.arange(0, 11) * 0.5
T4 = np.arange(1, 21) * 0.25
def up4(t10):
    w = np.concatenate([np.zeros((1, 2)), t10], 0)
    return np.stack([np.interp(T4, T2, w[:, d]) for d in range(2)], 1)

rng = np.random.default_rng(int(os.environ.get('SEL_SEED', '0')))
done = set()
for prev in ['selector_data.npz', 'selector_data2.npz']:
    if os.path.exists(prev):
        done |= set(np.load(prev, allow_pickle=True)['tokens'].tolist())
print(f'skipping {len(done)} already-collected scenes')
idx = rng.permutation(len(ds))
tokens, trajs, lps, rewards, ents, lpmins = [], [], [], [], [], []
with torch.no_grad():
    count = 0
    for i in idx:
        if count >= args.limit:
            break
        s = ds[int(i)]
        token = os.path.basename(s['data_path']).replace('.json', '')
        if token in done:
            continue
        count += 1
        prompt = (s['text'].split('<|im_start|>assistant')[0]
                  + '<|im_start|>assistant\n<answer>\nThe final output action is: ')
        inputs = processor(text=[prompt], videos=s['video_inputs'], padding=True,
                           return_tensors='pt').to('cuda')
        plen = inputs['input_ids'].shape[1]
        allowed = list(range(ASTART, ASTART + atok.n_bins))
        cand, cand_lp, cand_ent, cand_lpmin = [], [], [], []
        for k in range(args.K):
            out = vlm.generate(**inputs, max_new_tokens=10, do_sample=True,
                               temperature=1.0, top_p=0.95, output_scores=True,
                               return_dict_in_generate=True,
                               prefix_allowed_tokens_fn=lambda b, ids: allowed)
            gen = out.sequences[0, plen:][:10]
            steps = min(len(gen), len(out.scores))
            lps_t, ents_t = [], []
            for t in range(steps):
                logp = torch.log_softmax(out.scores[t][0].float(), -1)
                lps_t.append(logp[gen[t]].item())
                p = logp.exp()
                ents_t.append(-(p * logp.nan_to_num(neginf=0.0)).sum().item())
            lp = sum(lps_t)
            cand_ent.append(float(np.mean(ents_t)))
            cand_lpmin.append(float(np.min(lps_t)))
            if gen.shape[0] < 10:
                gen = torch.cat([gen, gen[-1:].repeat(10 - gen.shape[0])])
            tr = atok.decode_token_ids_to_trajectory(gen.cpu().long())
            cand.append(up4(np.asarray(tr)[0, 1:, :2])); cand_lp.append(lp)
        r = gt_rfs_reward(np.stack(cand), [token] * args.K)
        tokens.append(token); trajs.append(np.stack(cand))
        lps.append(cand_lp); rewards.append(r)
        ents.append(cand_ent); lpmins.append(cand_lpmin)
        if count % 100 == 0:
            print(f'{count}/{args.limit}', flush=True)
            np.savez(args.out, tokens=np.array(tokens), trajs=np.stack(trajs),
                     logprobs=np.array(lps), rewards=np.stack(rewards),
                     entropies=np.array(ents), lp_mins=np.array(lpmins))
np.savez(args.out, tokens=np.array(tokens), trajs=np.stack(trajs),
         logprobs=np.array(lps), rewards=np.stack(rewards),
         entropies=np.array(ents), lp_mins=np.array(lpmins))
print(f'saved {len(tokens)} scenes -> {args.out}')
