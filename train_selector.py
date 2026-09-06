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

# ---- train on selector_data (pseudo-RFS labels from training split) ----
z1 = np.load('selector_data.npz', allow_pickle=True)
z2 = np.load('selector_data2.npz', allow_pickle=True)
tokens = np.concatenate([z1['tokens'], z2['tokens']])
trajs = np.concatenate([z1['trajs'], z2['trajs']])
lps = np.concatenate([z1['logprobs'], z2['logprobs']])
rew = np.concatenate([z1['rewards'], z2['rewards']])
tb = pickle.load(open(f'{DATA}/gt_train.pkl', 'rb'))
sp = np.array([tb[t]['init_speed'] for t in tokens])
B, K = trajs.shape[:2]
X = features(trajs, lps, sp).reshape(B * K, -1)
y = rew.reshape(-1)
print(f'train scenes: {B} | samples: {len(y)}')
gbm = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                subsample=0.8, random_state=0)
gbm.fit(X, y)
print('train R2:', round(gbm.score(X, y), 3))

# ---- evaluate on 478 val (real 3-rater RFS) ----
zv = np.load('eval_bestofk_results.npz', allow_pickle=True)
names, vtrajs = list(zv['names']), zv['preds4hz']          # [B,8,20,2]
if 'logprobs' in zv.files:
    vlps = zv['logprobs']
else:
    print('WARNING: no logprobs in val npz - using zeros (logprob features dead)')
    vlps = np.zeros(vtrajs.shape[:2])
rated = {e['name']: e for e in pickle.load(open(f'{DATA}/rated_val.pkl', 'rb'))}
fr = [rated[n] for n in names]
past = np.stack([f['past'] for f in fr])
vsp = np.linalg.norm((past[:, -1] - past[:, -2]) / 0.25, axis=1)
Bv, Kv = vtrajs.shape[:2]
Xv = features(vtrajs, vlps, vsp).reshape(Bv * Kv, -1)
pred = gbm.predict(Xv).reshape(Bv, Kv)
pick = pred.argmax(1)

def rfs_of(traj):
    out = rfu.get_rater_feedback_score(
        inference_trajectories=traj[:, None, :, :], inference_probs=np.ones((Bv, 1)),
        rater_specified_trajectories=[f['raters'] for f in fr],
        rater_feedback_labels=[f['scores'] for f in fr], init_speed=vsp)
    return np.asarray(out['rater_feedback_score'])

per = np.stack([rfs_of(vtrajs[:, k]) for k in range(Kv)], 1)
sel = per[np.arange(Bv), pick]
pw = np.linalg.norm(vtrajs[:, :, None] - vtrajs[:, None, :], axis=-1).mean(-1)
med = per[np.arange(Bv), pw.sum(-1).argmin(1)]
print(f'\n=== selector on 478 val (real RFS) ===')
print(f'GBM-selector : {sel.mean():.3f}')
print(f'medoid       : {med.mean():.3f}')
print(f'oracle       : {per.max(1).mean():.3f}')
print(f'random       : {per.mean():.3f}')
imp = gbm.feature_importances_
fnames = ['pathlen','v_end','v_first','acc','curv','endpoint','lat_end',
          'centrality','cv_mismatch','lp','lp_rank','lp_z','grp_spread','v0']
top = np.argsort(imp)[::-1][:6]
print('top features:', [(fnames[i], round(imp[i], 3)) for i in top])
