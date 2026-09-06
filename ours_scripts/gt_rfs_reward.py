
import os, sys, json, glob, pickle
import numpy as np

DATA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DATA)
import rater_feedback_utils as rfu

CACHE = os.path.join(DATA, 'gt_train.pkl')
T2 = np.arange(0, 11) * 0.5
T4 = np.arange(0, 21) * 0.25   # includes t=0 -> 21 points, rater-style

def build_cache(json_dir):
    table = {}
    for f in glob.glob(os.path.join(json_dir, '*.json')):
        j = json.load(open(f))
        g = np.asarray(j['gt_trajectory'])[:, :2]                 # [10,2] @2Hz
        w = np.concatenate([np.zeros((1, 2)), g], 0)              # + origin
        g21 = np.stack([np.interp(T4, T2, w[:, d]) for d in range(2)], 1)  # [21,2] @4Hz
        v = np.asarray(j['velocity'])
        table[j['token']] = dict(gt21=g21.astype(np.float64),
                                 init_speed=float(np.linalg.norm(v)))
    with open(CACHE, 'wb') as fh:
        pickle.dump(table, fh)
    print(f'cached {len(table)} tokens -> {CACHE}')

_TABLE = None
def _table():
    global _TABLE
    if _TABLE is None:
        with open(CACHE, 'rb') as fh:
            _TABLE = pickle.load(fh)
    return _TABLE

def gt_rfs_reward(traj_4hz_20, tokens):
    """traj_4hz_20: [B,20,2] model trajectories @4Hz (t=0.25..5.0)
       tokens: list of B scene tokens -> returns [B] pseudo-RFS in [4,10]"""
    tb = _table()
    entries = [tb[t] for t in tokens]
    inference = np.asarray(traj_4hz_20, dtype=np.float64)[:, None, :, :]
    out = rfu.get_rater_feedback_score(
        inference_trajectories=inference,
        inference_probs=np.ones((len(tokens), 1)),
        rater_specified_trajectories=[[e['gt21']] for e in entries],
        rater_feedback_labels=[np.array([10.0]) for e in entries],
        init_speed=np.array([e['init_speed'] for e in entries]),
    )
    return np.asarray(out['rater_feedback_score'])

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', default='')
    args = ap.parse_args()
    if args.build:
        build_cache(args.build)
    # self-test on 50 random tokens
    tb = _table()
    toks = list(tb.keys())[:50]
    gt20 = np.stack([tb[t]['gt21'][1:] for t in toks])            # perfect prediction
    r_gt = gt_rfs_reward(gt20, toks)
    zeros = np.zeros_like(gt20)                                    # full-stop prediction
    r_zero = gt_rfs_reward(zeros, toks)
    # crude CV from first GT step (approximation for test only)
    v0 = gt20[:, 0, :] / 0.25
    cv = v0[:, None, :] * (np.arange(1, 21)[None, :, None] * 0.25)
    r_cv = gt_rfs_reward(cv, toks)
    print(f'reward(GT itself):  mean {r_gt.mean():.3f}  (expect ~10)')
    print(f'reward(CV-approx):  mean {r_cv.mean():.3f}  (expect mid/high)')
    print(f'reward(all zeros):  mean {r_zero.mean():.3f}  (expect low, ~4)')
