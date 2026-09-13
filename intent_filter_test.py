import sys, pickle, numpy as np, joblib
DATA = '/media/zara/6fbbe2fa-9fb0-49e6-bb18-67d7cd6c0e25/waymo_e2e'
sys.path.insert(0, DATA)
exec('def features' + open('train_selector.py').read().split('def features')[1].split('# ---- train')[0])
gbm = joblib.load('selector_gbm_merged.pkl')
rated = {e['name']: e for e in pickle.load(open(f'{DATA}/rated_val.pkl', 'rb'))}

# calibrate lateral sign convention from GT on turning scenes
ys_left, ys_right = [], []
for e in rated.values():
    y_end = e['future'][19][1] if len(e['future']) >= 20 else e['future'][-1][1]
    if e['intent'] == 2: ys_left.append(y_end)
    elif e['intent'] == 3: ys_right.append(y_end)
left_sign = np.sign(np.median(ys_left)) if ys_left else 1.0
print(f'calibration: intent=2 (left) n={len(ys_left)} median y={np.median(ys_left):+.1f} | '
      f'intent=3 (right) n={len(ys_right)} median y={np.median(ys_right):+.1f} | left_sign={left_sign:+.0f}')

Y_THR = 3.0
def intent_mask(vtrajs, intents):
    """True = keep. Drop candidates clearly turning the wrong way on turn scenes."""
    B, K = vtrajs.shape[:2]
    keep = np.ones((B, K), dtype=bool)
    for b in range(B):
        it = intents[b]
        if it not in (2, 3): continue
        want = left_sign if it == 2 else -left_sign
        y_end = vtrajs[b, :, -1, 1]
        wrong = (np.sign(y_end) == -want) & (np.abs(y_end) > Y_THR)
        if wrong.all(): continue                      # fail-safe: keep pool
        keep[b] = ~wrong
    return keep

def run(npz_path, label):
    zv = np.load(npz_path, allow_pickle=True)
    names, vtrajs, vlps, per = list(zv['names']), zv['preds4hz'], zv['logprobs'], zv['per_cand']
    fr = [rated[n] for n in names]
    intents = [f['intent'] for f in fr]
    past = np.stack([f['past'] for f in fr])
    vsp = np.linalg.norm((past[:, -1] - past[:, -2]) / 0.25, axis=1)
    B, K = vtrajs.shape[:2]
    keep = intent_mask(vtrajs, intents)
    n_drop = (~keep).sum()
    scenes_hit = (~keep).any(1).sum()
    pred = gbm.predict(features(vtrajs, vlps, vsp).reshape(B*K, -1)).reshape(B, K)
    pw = np.linalg.norm(vtrajs[:, :, None] - vtrajs[:, None, :], axis=-1).mean(-1)
    def medoid_sel(mask):
        out = []
        for b in range(B):
            idxs = np.where(mask[b])[0]
            sub = pw[b][np.ix_(idxs, idxs)]
            out.append(per[b, idxs[sub.sum(-1).argmin()]])
        return np.mean(out)
    def gbm_sel(mask):
        p = np.where(mask, pred, -np.inf)
        return per[np.arange(B), p.argmax(1)].mean()
    print(f'\n--- {label} ---')
    print(f'dropped {n_drop} candidates across {scenes_hit} scenes '
          f'({(np.array(intents)==2).sum()} left / {(np.array(intents)==3).sum()} right scenes total)')
    print(f'medoid : {medoid_sel(np.ones_like(keep)):.3f} -> {medoid_sel(keep):.3f}')
    print(f'gbm    : {gbm_sel(np.ones_like(keep)):.3f} -> {gbm_sel(keep):.3f}')
    print(f'oracle : {per.max(1).mean():.3f}')

run('eval_bestofk8_backup.npz', 'batch 1 (K=8)')
run('eval_bestofk_results.npz', 'batch 2 (K=8)')
