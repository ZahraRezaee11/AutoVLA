import glob, os, time, lmdb
import tensorflow as tf
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2

DATA = '/media/zara/6fbbe2fa-9fb0-49e6-bb18-67d7cd6c0e25/waymo_e2e'
LMDB_DIR = os.path.join(DATA, 'training_lmdb')
files = sorted(glob.glob(f'{DATA}/training_*.tfrecord-*'))
print(f'{len(files)} shards')

env = lmdb.open(LMDB_DIR, map_size=100 * 1024**3)
n, t0 = 0, time.time()
txn = env.begin(write=True)
for fi, fp in enumerate(files):
    for rec in tf.data.TFRecordDataset(fp, compression_type='').as_numpy_iterator():
        d = wod_e2ed_pb2.E2EDFrame()
        d.ParseFromString(rec)
        token = d.frame.context.name
        d.frame.ClearField('images')
        txn.put(token.encode('utf-8'), d.SerializeToString())
        n += 1
        if n % 500 == 0:
            txn.commit(); txn = env.begin(write=True)
    print(f'shard {fi+1}/{len(files)} | {n} records | {n/(time.time()-t0):.0f} rec/s', flush=True)
txn.commit(); env.sync(); env.close()
print(f'DONE: {n} records in {(time.time()-t0)/60:.1f} min')
