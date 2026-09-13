import os, sys, pickle, argparse, joblib
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.abspath('./navsim'))
DATA = '/media/zara/6fbbe2fa-9fb0-49e6-bb18-67d7cd6c0e25/waymo_e2e'
sys.path.insert(0, DATA)
import yaml, torch, numpy as np
from transformers import AutoProcessor
from peft import get_peft_model, LoraConfig, TaskType
from dataset_utils.sft_dataset import SFTDataset
from models.autovla import SFTAutoVLA
import rater_feedback_utils as rfu
exec('def features' + open('train_selector.py').read().split('def features')[1].split('# ---- train')[0])

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True)
p.add_argument('--config', default='config/training/waymo-rft-v3.yaml')
p.add_argument('--limit', type=int, default=0)
p.add_argument('--rounds', type=int, default=2)
p.add_argument('--K', type=int, default=8)
args = p.parse_args()

config = yaml.safe_load(open(args.config)); mc = config['model']
processor = AutoProcessor.from_pretrained(mc['pretrained_model_path'], use_fast=True)
ds = SFTDataset({'json_dataset_path': './dataset/waymo_val_rated', 'sensor_data_path': None},
                mc, processor, using_cot=False)
ASTART = mc['tokens']['action_start_id']; atok = ds.action_tokenizer
model = SFTAutoVLA(config); lc = mc['lora']
model.autovla.vlm = get_peft_model(model.autovla.vlm, LoraConfig(
    modules_to_save=["embed_tokens", "lm_head"], task_type=TaskType['CAUSAL_LM'],
    target_modules=lc['target_modules'], r=lc['r'], lora_alpha=lc['alpha'], lora_dropout=lc['dropout']))
sd = torch.load(args.ckpt, map_location='cpu')['state_dict']
sd = {(k[len('model.'):] if k.startswith('model.') else k): v for k, v in sd.items()}
print('load:', [len(x) for x in model.load_state_dict(sd, strict=False)])
vlm = model.autovla.vlm.eval().to('cuda', dtype=torch.bfloat16)
gbm = joblib.load('selector_gbm_merged.pkl')
rated = {e['name']: e for e in pickle.load(open(f'{DATA}/rated_val.pkl', 'rb'))}

T2 = np.arange(0, 11) * 0.5; T4 = np.arange(1, 21) * 0.25
def up4(t10):
    w = np.concatenate([np.zeros((1, 2)), t10], 0)
    return np.stack([np.interp(T4, T2, w[:, d]) for d in range(2)], 1)
ALLOWED = list(range(ASTART, ASTART + atok.n_bins))

def sample_one(inputs, plen, prefix=None):
    """prefix: list of token ids to lock at positions 0..len-1"""
    def allow(batch_id, ids):
        t = ids.shape[0] - plen
        if prefix is not None and t < len(prefix):
            return [prefix[t]]
        return ALLOWED
    out = vlm.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=1.0, top_p=0.95,
                       output_scores=True, return_dict_in_generate=True, prefix_allowed_tokens_fn=allow)
    gen = out.sequences[0, plen:][:10]
    lp = sum(torch.log_softmax(out.scores[t][0].float(), -1)[gen[t]].item()
             for t in range(min(len(gen), len(out.scores))))
    if gen.shape[0] < 10: gen = torch.cat([gen, gen[-1:].repeat(10 - gen.shape[0])])
    tr = atok.decode_token_ids_to_trajectory(gen.cpu().long())
    return [int(t) for t in gen], up4(np.asarray(tr)[0, 1:, :2]), lp

def gbm_scores(trajs, lps, sp):
    X = features(np.asarray(trajs)[None], np.asarray(lps)[None], np.array([sp]))
    return gbm.predict(X.reshape(len(trajs), -1))

rng = np.random.default_rng(0)
items = list(range(len(ds)))[:args.limit or None]
names, pick_trs, pick_ind, all_cands = [], [], [], []
with torch.no_grad():
    for i in items:
        s = ds[i]; name = os.path.basename(s['data_path']).replace('.json', '')
        past = rated[name]['past']; sp = float(np.linalg.norm((past[-1] - past[-2]) / 0.25))
        prompt = s['text'].split('<|im_start|>assistant')[0] + '<|im_start|>assistant\n<answer>\nThe final output action is: '
        inputs = processor(text=[prompt], videos=s['video_inputs'], padding=True, return_tensors='pt').to('cuda')
        plen = inputs['input_ids'].shape[1]
        # round 0: independent
        toks, trajs, lps = zip(*[sample_one(inputs, plen) for _ in range(args.K)])
        toks, trajs, lps = list(toks), list(trajs), list(lps)
        sc = gbm_scores(trajs, lps, sp); c = int(sc.argmax()); c_score = sc[c]
        region = 4   # locked-prefix length; larger = tighter region
        for r in range(args.rounds):
            new = [sample_one(inputs, plen, prefix=toks[c][:max(1, region + rng.integers(-1, 2))]) for _ in range(args.K)]
            nt, ntr, nlp = zip(*new)
            toks += list(nt); trajs += list(ntr); lps += list(nlp)
            sc = gbm_scores(trajs, lps, sp)
            best = int(sc.argmax())
            if sc[best] > c_score + 1e-6: c, c_score, region = best, sc[best], max(2, region - 1)
            else: region = min(8, region + 2)
        names.append(name); pick_trs.append(trajs[c]); all_cands.append(np.stack(trajs))
        # independent baseline with equal budget: GBM over the round-0 first K only
        sc0 = gbm_scores(trajs[:args.K], lps[:args.K], sp); pick_ind.append(trajs[int(sc0.argmax())])
        if len(names) % 25 == 0: print(f'{len(names)}/{len(items)}', flush=True)

fr = [rated[n] for n in names]; past = np.stack([f['past'] for f in fr])
vsp = np.linalg.norm((past[:, -1] - past[:, -2]) / 0.25, axis=1); B = len(fr)
def rfs_of(tr):
    out = rfu.get_rater_feedback_score(inference_trajectories=np.asarray(tr)[:, None], inference_probs=np.ones((B, 1)),
        rater_specified_trajectories=[f['raters'] for f in fr], rater_feedback_labels=[f['scores'] for f in fr], init_speed=vsp)
    return np.asarray(out['rater_feedback_score'])
Kall = all_cands[0].shape[0]
per = np.stack([rfs_of([c[k] for c in all_cands]) for k in range(Kall)], 1)
print(f'\n=== TRS on {B} frames | K={args.K} x (1+{args.rounds}) rounds = {Kall} candidates ===')
print(f'GBM pick, independent K={args.K}   : {rfs_of(pick_ind).mean():.3f}')
print(f'GBM pick, TRS search           : {rfs_of(pick_trs).mean():.3f}')
print(f'oracle over all {Kall} candidates  : {per.max(1).mean():.3f}')
print(f'oracle over round-0 only       : {per[:, :args.K].max(1).mean():.3f}')
np.savez('eval_trs_results.npz', names=np.array(names), per_cand=per, cands=np.stack(all_cands))

