import os, sys, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml
from transformers import AutoProcessor
from dataset_utils.preprocessing.waymo_e2e_dataset import WaymoE2ECoTAnnotationDataset
from tools.preprocessing.nocot_sample_generation import process_sample

RATED = pickle.load(open('/media/zara/6fbbe2fa-9fb0-49e6-bb18-67d7cd6c0e25/waymo_e2e/rated_val.pkl','rb'))
rated_tokens = [e['name'] for e in RATED]
print(f"{len(rated_tokens)} rated tokens")

class RatedEvalDataset(WaymoE2ECoTAnnotationDataset):
    def scene_loader(self):
        cfg = self.config
        images_root = os.path.join(cfg['dataset_path'], self.split + '_images')
        fr = int(cfg['raw_images_freq'] / cfg['model_freq'])
        nh = fr * (cfg['model_his_frames'] - 1) + 1
        scenes, missing = {}, 0
        for tok in rated_tokens:
            seq, idx = tok.rsplit('-', 1); cur = int(idx)
            seq_dir = os.path.join(images_root, seq)
            frame_sequence = [cur - m for m in range(nh - 1, -1, -fr)]
            entry = self._make_scene_entry(seq, seq_dir, frame_sequence)
            if entry: scenes[tok] = entry
            else: missing += 1
        print(f"built {len(scenes)} eval scenes, {missing} missing")
        return list(scenes.items())

cfg = yaml.safe_load(open('config/dataset/waymo-val-ours.yaml'))
proc = AutoProcessor.from_pretrained(cfg['pretrained_model_path'], use_fast=True)
ds = RatedEvalDataset(cfg, proc)

outdir = './dataset/waymo_val_rated'
os.makedirs(outdir, exist_ok=True)
for i in range(len(ds)):
    token, result = process_sample(ds[i], 'waymo')
    json.dump(result, open(os.path.join(outdir, f'{token}.json'), 'w'))
    if (i+1) % 50 == 0: print(f'{i+1}/{len(ds)}', flush=True)
print("DONE")
