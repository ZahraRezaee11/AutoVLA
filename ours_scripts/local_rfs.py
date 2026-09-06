import glob, os, pickle, numpy as np

DATA = '/media/zara/6fbbe2fa-9fb0-49e6-bb18-67d7cd6c0e25/waymo_e2e'
CACHE = 'rated_val.pkl'
DT, T = 0.25, 20

# ---------- phase 1: cache rated frames ----------
if not os.path.exists(CACHE):
    import tensorflow as tf
    from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
    frames = []
    files = sorted(glob.glob(f'{DATA}/val_*.tfrecord*'))
    for fi, fp in enumerate(files):
        for rec in tf.data.TFRecordDataset(fp).as_numpy_iterator():
            d = wod_e2ed_pb2.E2EDFrame(); d.ParseFromString(rec)
            if not (len(d.preference_trajectories) and
                    d.preference_trajectories[0].preference_score != -1):
                continue
            frames.append(dict(
                name=d.frame.context.name,
                intent=d.intent,
                past=np.stack([d.past_states.pos_x, d.past_states.pos_y], 1),
                future=np.stack([d.future_states.pos_x, d.future_states.pos_y], 1),
                raters=[np.stack([p.pos_x, p.pos_y], 1).astype(np.float64)
                        for p in d.preference_trajectories],
                scores=np.array([p.preference_score for p in d.preference_trajectories],
                                dtype=np.float64),
            ))
        print(f'shard {fi+1}/{len(files)} | rated so far: {len(frames)}', flush=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(frames, f)
    print('cached ->', CACHE)

# ---------- phase 2: metric ----------
import rater_feedback_utils as rfu
with open(CACHE, 'rb') as f:
    frames = pickle.load(f)
B = len(frames)
print(f'{B} rated frames')
print('example rater lengths (frame 0):', [len(r) for r in frames[0]['raters']])

past = np.stack([fr['past'] for fr in frames])          # [B, 16, 2]

def cv_pred(p):
    v = (p[-1] - p[-2]) / DT
    t = np.arange(1, T + 1)[:, None] * DT
    return p[-1] + v[None, :] * t

preds = np.stack([cv_pred(p) for p in past])            # [B, 20, 2]
inference = preds[:, None, :, :]                        # [B, K=1, 20, 2]
probs = np.ones((B, 1))
init_speed = np.linalg.norm((past[:, -1] - past[:, -2]) / DT, axis=1)

rater_trajs = [fr['raters'] for fr in frames]           # List[List[np.ndarray]] var-len
rater_labels = [fr['scores'] for fr in frames]          # List[np.ndarray [3]]

out = rfu.get_rater_feedback_score(
    inference_trajectories=inference,
    inference_probs=probs,
    rater_specified_trajectories=rater_trajs,
    rater_feedback_labels=rater_labels,
    init_speed=init_speed,
)
print('keys:', list(out.keys()))
for k, v in out.items():
    print(k, np.asarray(v).shape)
rfs = np.asarray(out['rater_feedback_score']) if 'rater_feedback_score' in out else None
if rfs is not None:
    print(f'\n=== constant-velocity baseline ===')
    print(f'RFS mean: {rfs.mean():.3f}')
    print(f'RFS distribution: min {rfs.min():.1f} | p25 {np.percentile(rfs,25):.1f} | '
          f'median {np.median(rfs):.1f} | p75 {np.percentile(rfs,75):.1f} | max {rfs.max():.1f}')
    print(f'fraction at floor (<=4.0): {(rfs <= 4.0 + 1e-6).mean()*100:.1f}%')
    print(f'fraction >= 8: {(rfs >= 8).mean()*100:.1f}%')