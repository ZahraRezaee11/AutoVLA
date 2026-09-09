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



def features_ext(trajs, lps, sp, ents, lpm):
    base = features(trajs, lps, sp)
    ent_z = (ents - ents.mean(-1, keepdims=True)) / (ents.std(-1, keepdims=True) + 1e-6)
    lpm_z = (lpm - lpm.mean(-1, keepdims=True)) / (lpm.std(-1, keepdims=True) + 1e-6)
    return np.concatenate([base, ents[..., None], ent_z[..., None],
                           lpm[..., None], lpm_z[..., None]], -1)

# entropy features exist only in data3 -> train extended model on data3 alone,
# plus a merged old-feature model; compare both on the fresh val npz
z3 = np.load('selector_data3.npz', allow_pickle=True)
trajs, lps, rew = z3['trajs'].astype(np.float32), z3['logprobs'], z3['rewards']
ents, lpm = z3['entropies'], z3['lp_mins']
tb = pickle.load(open(f'{DATA}/gt_train.pkl', 'rb'))
sp = np.array([tb[t]['init_speed'] for t in z3['tokens']])
B, K = trajs.shape[:2]
print(f'data3 scenes: {B}')

Xe = features_ext(trajs, lps, sp, ents, lpm).reshape(B*K, -1)
y = rew.reshape(-1)
gbm_e = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, random_state=0)
gbm_e.fit(Xe, y)
print('R2 (data3, ext features):', round(gbm_e.score(Xe, y), 3))

import joblib
gbm_m = joblib.load('selector_gbm_merged.pkl')

zv = np.load('eval_bestofk_results.npz', allow_pickle=True)
names, vtrajs, vlps = list(zv['names']), zv['preds4hz'], zv['logprobs']
vents, vlpm = zv['entropies'], zv['lp_mins']
per = zv['per_cand']
rated = {e['name']: e for e in pickle.load(open(f'{DATA}/rated_val.pkl', 'rb'))}
fr = [rated[n] for n in names]
past = np.stack([f['past'] for f in fr])
vsp = np.linalg.norm((past[:, -1] - past[:, -2]) / 0.25, axis=1)
Bv, Kv = vtrajs.shape[:2]

pred_m = gbm_m.predict(features(vtrajs, vlps, vsp).reshape(Bv*Kv, -1)).reshape(Bv, Kv)
pred_e = gbm_e.predict(features_ext(vtrajs, vlps, vsp, vents, vlpm).reshape(Bv*Kv, -1)).reshape(Bv, Kv)
pred_c = 0.5 * ((pred_m - pred_m.mean(1, keepdims=True)) / (pred_m.std(1, keepdims=True) + 1e-6)
              + (pred_e - pred_e.mean(1, keepdims=True)) / (pred_e.std(1, keepdims=True) + 1e-6))

for name, pr in [('merged-old-feats', pred_m), ('data3-ext-feats', pred_e), ('combo', pred_c)]:
    sel = per[np.arange(Bv), pr.argmax(1)]
    print(f'{name:18s}: {sel.mean():.3f}')
pw = np.linalg.norm(vtrajs[:, :, None] - vtrajs[:, None, :], axis=-1).mean(-1)
print(f'{"medoid":18s}: {per[np.arange(Bv), pw.sum(-1).argmin(1)].mean():.3f}')
print(f'{"oracle":18s}: {per.max(1).mean():.3f}')
fn = ['pathlen','v_end','v_first','acc','curv','endpoint','lat_end','centrality',
      'cv_mismatch','lp','lp_rank','lp_z','grp_spread','v0','ent','ent_z','lpmin','lpmin_z']
imp = gbm_e.feature_importances_
print('top ext features:', [(fn[i], round(imp[i],3)) for i in np.argsort(imp)[::-1][:6]])
