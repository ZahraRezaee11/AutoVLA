import sys, pickle, numpy as np
DATA = '/media/zara/6fbbe2fa-9fb0-49e6-bb18-67d7cd6c0e25/waymo_e2e'
sys.path.insert(0, DATA)
import rater_feedback_utils as rfu
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
exec(open('train_selector.py').read().split('# ---- evaluate')[0])   # loads gbm + features + train data

# pairwise ranker: classify which of two candidates from same scene is better
Bt, Kt = trajs.shape[:2]
Xf = features(trajs, lps, sp)                                        # [B,K,F]
rng = np.random.default_rng(0)
pairs_X, pairs_y = [], []
for b in range(Bt):
    for _ in range(12):
        i, j = rng.choice(Kt, 2, replace=False)
        if abs(rew[b, i] - rew[b, j]) < 0.3: continue
        pairs_X.append(Xf[b, i] - Xf[b, j])
        pairs_y.append(1.0 if rew[b, i] > rew[b, j] else 0.0)
pairs_X, pairs_y = np.asarray(pairs_X), np.asarray(pairs_y)
print(f'pairs: {len(pairs_y)}')
rk = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                subsample=0.8, random_state=0)
rk.fit(pairs_X, pairs_y)
print('pair acc (train):', round(rk.score(pairs_X, pairs_y), 3))

def evaluate(npz_path, label):
    zv = np.load(npz_path, allow_pickle=True)
    names, vtrajs, vlps = list(zv['names']), zv['preds4hz'], zv['logprobs']
    rated = {e['name']: e for e in pickle.load(open(f'{DATA}/rated_val.pkl', 'rb'))}
    fr = [rated[n] for n in names]
    past = np.stack([f['past'] for f in fr])
    vsp = np.linalg.norm((past[:, -1] - past[:, -2]) / 0.25, axis=1)
    Bv, Kv = vtrajs.shape[:2]
    Xv = features(vtrajs, vlps, vsp)                                  # [B,K,F]
    per = zv['per_cand'] if 'per_cand' in zv.files else None
    if per is None:
        def rfs_of(traj):
            out = rfu.get_rater_feedback_score(
                inference_trajectories=traj[:, None, :, :], inference_probs=np.ones((Bv, 1)),
                rater_specified_trajectories=[f['raters'] for f in fr],
                rater_feedback_labels=[f['scores'] for f in fr], init_speed=vsp)
            return np.asarray(out['rater_feedback_score'])
        per = np.stack([rfs_of(vtrajs[:, k]) for k in range(Kv)], 1)
    pred = gbm.predict(Xv.reshape(Bv * Kv, -1)).reshape(Bv, Kv)
    pw = np.linalg.norm(vtrajs[:, :, None] - vtrajs[:, None, :], axis=-1).mean(-1)
    print(f'\n--- {label} (K={Kv}) ---')
    print(f'gbm argmax        : {per[np.arange(Bv), pred.argmax(1)].mean():.3f}')
    for M in [3, 4, 6]:
        if M >= Kv: continue
        sel = []
        for b in range(Bv):
            top = np.argsort(pred[b])[::-1][:M]
            sub = pw[b][np.ix_(top, top)]
            sel.append(per[b, top[sub.sum(-1).argmin()]])
        print(f'top-{M} -> medoid   : {np.mean(sel):.3f}')
    # pairwise ranker: Borda count (wins against all others)
    sel_rk = []
    for b in range(Bv):
        wins = np.zeros(Kv)
        for i in range(Kv):
            diffs = Xv[b, i][None].repeat(Kv, 0) - Xv[b]
            wins[i] = rk.predict_proba(diffs)[:, 1].sum()
        sel_rk.append(per[b, wins.argmax()])
    print(f'pairwise ranker   : {np.mean(sel_rk):.3f}')
    print(f'medoid            : {per[np.arange(Bv), pw.sum(-1).argmin(1)].mean():.3f}')
    print(f'oracle            : {per.max(1).mean():.3f}')

evaluate('eval_bestofk8_backup.npz', 'K=8')
evaluate('eval_bestofk_results.npz', 'K=16')
