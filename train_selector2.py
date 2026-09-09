import sys, pickle, numpy as np
DATA = '/media/zara/6fbbe2fa-9fb0-49e6-bb18-67d7cd6c0e25/waymo_e2e'
sys.path.insert(0, DATA)
import rater_feedback_utils as rfu
from sklearn.ensemble import GradientBoostingRegressor
def features(trajs, lps, sp):
    """trajs [B,K,20,2], lps [B,K], sp [B] -> feats [B,K,F]"""
    B, K = trajs.shape[:2]
    v0 = sp[:, None]
    step = np.diff(np.concatenate([np.zeros((B, K, 1, 2)), trajs], 2), axis=2)  # [B,K,20,2]
    seg = np.linalg.norm(step, axis=-1)                    # [B,K,20]
    pathlen = seg.sum(-1)
    speed = seg / 0.25
    v_end = speed[..., -1]
    v_first = speed[..., 0]
    acc = np.abs(np.diff(speed, axis=-1)).max(-1)
    heading = np.arctan2(step[..., 1], step[..., 0])
    curv = np.abs(np.diff(heading, axis=-1)).sum(-1)
    endpoint = np.linalg.norm(trajs[:, :, -1], axis=-1)
    lat_end = np.abs(trajs[:, :, -1, 1])
    pw = np.linalg.norm(trajs[:, :, None] - trajs[:, None, :], axis=-1).mean(-1)  # [B,K,K]
    centrality = pw.mean(-1)
    cv_end = v0 * 5.0
    cv_mismatch = np.abs(endpoint - cv_end)
    lp = np.asarray(lps)
    lp_rank = lp.argsort(-1).argsort(-1).astype(float)
    lp_z = (lp - lp.mean(-1, keepdims=True)) / (lp.std(-1, keepdims=True) + 1e-6)
    grp_spread = pw.mean((-1, -2))[:, None].repeat(K, 1)
    v0_f = v0.repeat(K, 1)
    feats = np.stack([pathlen, v_end, v_first, acc, curv, endpoint, lat_end,
                      centrality, cv_mismatch, lp, lp_rank, lp_z, grp_spread, v0_f], -1)
    return feats



paths = ['selector_data.npz', 'selector_data2.npz', 'selector_data3.npz']
seen = set()
T, TR, L, R = [], [], [], []
for p in paths:
    z = np.load(p, allow_pickle=True, mmap_mode='r')
    toks = z['tokens']
    keep = [i for i, t in enumerate(toks) if t not in seen]
    seen.update(toks[keep])
    T.append(toks[keep]); TR.append(z['trajs'][keep])
    L.append(z['logprobs'][keep]); R.append(z['rewards'][keep])
    del z
tokens = np.concatenate(T)
trajs = np.concatenate(TR).astype(np.float32)
lps = np.concatenate(L); rew = np.concatenate(R)
del T, TR, L, R
tb = pickle.load(open(f'{DATA}/gt_train.pkl', 'rb'))
sp = np.array([tb[t]['init_speed'] for t in tokens])
B, K = trajs.shape[:2]
print(f'merged scenes (dedup): {B} | samples: {B*K}')

X = features(trajs, lps, sp).reshape(B*K, -1)
y = rew.reshape(-1)
gbm = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                                subsample=0.8, random_state=0)
gbm.fit(X, y)
print('R2 (merged, old features):', round(gbm.score(X, y), 3))

zv = np.load('eval_bestofk8_backup.npz', allow_pickle=True)
names, vtrajs, vlps = list(zv['names']), zv['preds4hz'], zv['logprobs']
rated = {e['name']: e for e in pickle.load(open(f'{DATA}/rated_val.pkl', 'rb'))}
fr = [rated[n] for n in names]
past = np.stack([f['past'] for f in fr])
vsp = np.linalg.norm((past[:, -1] - past[:, -2]) / 0.25, axis=1)
Bv, Kv = vtrajs.shape[:2]
per = zv['per_cand']
pred = gbm.predict(features(vtrajs, vlps, vsp).reshape(Bv*Kv, -1)).reshape(Bv, Kv)
sel = per[np.arange(Bv), pred.argmax(1)]
print(f'GBM (merged scenes, old features) on K=8 val: {sel.mean():.3f}   [prev 4.3k: 7.483]')
print(f'oracle: {per.max(1).mean():.3f}')
import joblib; joblib.dump(gbm, 'selector_gbm_merged.pkl')
print('saved selector_gbm_merged.pkl')
