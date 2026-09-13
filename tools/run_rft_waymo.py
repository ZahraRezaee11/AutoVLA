
import os, sys, argparse, random
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./navsim'))
DATA = '/media/zara/6fbbe2fa-9fb0-49e6-bb18-67d7cd6c0e25/waymo_e2e'
sys.path.insert(0, DATA)

import yaml, torch, numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from peft import get_peft_model, LoraConfig, TaskType

from dataset_utils.sft_dataset import SFTDataset
from models.autovla import SFTAutoVLA
from gt_rfs_reward import gt_rfs_reward

T2 = np.arange(0, 11) * 0.5
T4 = np.arange(1, 21) * 0.25

def up4hz(traj10):
    w = np.concatenate([np.zeros((1, 2)), traj10], 0)
    return np.stack([np.interp(T4, T2, w[:, d]) for d in range(2)], 1)

class WaymoGRPO(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        mc = config['model']
        self.automatic_optimization = False
        self.processor = AutoProcessor.from_pretrained(mc['pretrained_model_path'], use_fast=True)
        base = SFTAutoVLA(config)
        lc = mc['lora']
        base.autovla.vlm = get_peft_model(base.autovla.vlm, LoraConfig(
            modules_to_save=["embed_tokens", "lm_head"],
            task_type=TaskType[lc.get('task_type', 'CAUSAL_LM')],
            target_modules=lc['target_modules'], r=lc['r'],
            lora_alpha=lc['alpha'], lora_dropout=lc['dropout'], bias=lc.get('bias', 'none')))
        sd = torch.load(mc['sft_model_path'], map_location='cpu')['state_dict']
        sd = { (k[len('model.'):] if k.startswith('model.') else k): v for k, v in sd.items() }
        missing, unexpected = base.load_state_dict(sd, strict=False)
        print(f'SFT ckpt loaded: {len(missing)} missing, {len(unexpected)} unexpected')
        self.model = base
        self.astart = mc['tokens']['action_start_id']
        rl = config['rl']
        self.G = rl['group_size']
        self.G_backward = rl.get('backward_size', rl['group_size'])
        self.temperature = rl['temperature']
        self.top_p = rl['top_p']
        self.lr = rl['lr']
        self.skip_counts = {}
        self.skip_threshold = rl.get('skip_after_degenerate', 2)

    def train_dataloader(self):
        ds = SFTDataset({'json_dataset_path': self.cfg['data']['train']['json_dataset_path'],
                         'sensor_data_path': None},
                        self.cfg['model'], self.processor, using_cot=False)
        hard_path = os.environ.get('HARD_TOKENS', '')
        if hard_path:
            hard = set(np.load(hard_path, allow_pickle=True).tolist())
            n0 = len(ds.scenes)
            ds.scenes = [sc for sc in ds.scenes if sc[0].stem in hard]
            print(f'curriculum: filtered scenes {n0} -> {len(ds.scenes)}')
            assert len(ds.scenes) > 0, 'hard-token filter removed everything'
        self.action_tokenizer = ds.action_tokenizer
        self.n_bins = ds.action_tokenizer.n_bins
        return DataLoader(ds, batch_size=1, shuffle=True, num_workers=2,
                          collate_fn=lambda x: x[0])

    def training_step(self, sample, batch_idx):
        opt = self.optimizers()
        vlm = self.model.autovla.vlm
        token = os.path.basename(sample['data_path']).replace('.json', '')
        if self.skip_counts.get(token, 0) >= self.skip_threshold:
            return None
        prompt = (sample['text'].split('<|im_start|>assistant')[0]
                  + '<|im_start|>assistant\n<answer>\nThe final output action is: ')
        inputs = self.processor(text=[prompt], videos=sample['video_inputs'],
                                padding=True, return_tensors='pt').to(self.device)
        plen = inputs['input_ids'].shape[1]
        allowed = list(range(self.astart, self.astart + self.n_bins))

        # --- G rollouts (no grad) ---
        seqs, trajs = [], []
        with torch.no_grad():
            for g in range(self.G):
                torch.manual_seed(self.global_step * 1000 + g)
                out = vlm.generate(**inputs, max_new_tokens=10, do_sample=True,
                                   temperature=self.temperature, top_p=self.top_p,
                                   prefix_allowed_tokens_fn=lambda b, ids: allowed)
                comp = out[0, plen:][:10]
                if comp.shape[0] < 10:
                    comp = torch.cat([comp, comp[-1:].repeat(10 - comp.shape[0])])
                seqs.append(comp)
                _prev = torch.get_default_dtype()
                torch.set_default_dtype(torch.float32)
                tr = self.action_tokenizer.decode_token_ids_to_trajectory(comp.cpu().long())
                torch.set_default_dtype(_prev)
                trajs.append(up4hz(np.asarray(tr)[0, 1:, :2]))

        rewards = gt_rfs_reward(np.stack(trajs), [token] * self.G)
        r = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        std = r.std()
        self.log('reward_mean', r.mean(), prog_bar=True, on_step=True)
        self.log('reward_std', std, prog_bar=True, on_step=True)
        if std < 1e-4:
            self.skip_counts[token] = self.skip_counts.get(token, 0) + 1
            self.log('skipped_scenes', float(len([v for v in self.skip_counts.values() if v >= self.skip_threshold])), on_step=True)
            return None  # no learning signal in this group
        self.skip_counts[token] = 0
        adv = (r - r.mean()) / (std + 1e-4)

        # --- policy gradient over the group ---
        opt.zero_grad()
        order = torch.argsort(adv.abs(), descending=True)[:self.G_backward].tolist()
        for g in order:
            ids = torch.cat([inputs['input_ids'][0], seqs[g]]).unsqueeze(0)
            att = torch.ones_like(ids)
            logits = vlm(input_ids=ids, attention_mask=att,
                         pixel_values_videos=inputs['pixel_values_videos'],
                         video_grid_thw=inputs['video_grid_thw']).logits
            lp = torch.log_softmax(logits[:, plen-1:-1, :], dim=-1)
            tok_lp = lp.gather(2, seqs[g].unsqueeze(0).unsqueeze(-1)).squeeze(-1)
            loss = -(adv[g] * tok_lp.mean()) / len(order)
            self.manual_backward(loss)
        opt.step()
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
        self.log('adv_max', adv.max(), on_step=True)
        return None

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg['rl']['max_steps'], eta_min=self.lr * 0.1)
        return [opt], [{'scheduler': sched, 'interval': 'step'}]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--max_steps', type=int, default=0)
    args = ap.parse_args()
    config = yaml.safe_load(open(f'config/{args.config}.yaml'))
    steps = args.max_steps or config['rl']['max_steps']
    model = WaymoGRPO(config)
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=f"runs/rft/{config['name']}", every_n_train_steps=250,
        save_top_k=2, monitor='reward_mean', mode='max',
        filename='step{step}-r{reward_mean:.3f}')
    trainer = pl.Trainer(max_steps=steps, accelerator='gpu', devices=1,
                         precision='bf16-true', strategy='auto',
                         callbacks=[ckpt_cb], logger=True,
                         enable_checkpointing=True, log_every_n_steps=5)
    trainer.fit(model)

if __name__ == '__main__':
    main()
