import os
import zipfile

import torch
import wandb
import lightning.pytorch as pl


class StateActionEffectDM(pl.LightningDataModule):
    def __init__(self, name, batch_size=32, num_workers=0, obj_relative=False, n=0):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.obj_relative = obj_relative
        self.n = n

    def prepare_data(self):
        self.data_path = os.path.join("data", self.name)
        artifact = wandb.use_artifact(f"{wandb.api.default_entity}/attentive-deepsym/{self.name}:latest",
                                      type="dataset")
        if not os.path.exists(self.data_path):
            artifact.download(root=self.data_path)
            archive = zipfile.ZipFile(os.path.join(self.data_path, f"{self.name}.zip"), "r")
            archive.extractall(self.data_path)
            archive.close()
            os.remove(os.path.join(self.data_path, f"{self.name}.zip"))
        self.train_set = StateActionEffectDataset(self.name, split="train", obj_relative=self.obj_relative, n=self.n)
        self.val_set = StateActionEffectDataset(self.name, split="val", obj_relative=self.obj_relative)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                           num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size,
                                           num_workers=self.num_workers)


class StateActionEffectDataset(torch.utils.data.Dataset):
    def __init__(self, name, split="train", obj_relative=False, n=0):
        path = os.path.join("data", name)
        self.obj_relative = obj_relative
        self.state = torch.load(os.path.join(path, "state.pt"))
        self.action = torch.load(os.path.join(path, "action.pt"))
        self.effect = torch.load(os.path.join(path, "effect.pt"))
        self.mask = torch.load(os.path.join(path, "mask.pt"))
        self.post_state = torch.load(os.path.join(path, "post_state.pt"))
        n_train = int(len(self.state) * 0.8)
        n_val = int(len(self.state) * 0.1)
        self.binary = torch.tensor([[0, 0, 0, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 1, 0, 0],
                                    [1, 0, 0, 0]])
        if split == "train":
            self.state = self.state[:n_train]
            self.action = self.action[:n_train]
            self.effect = self.effect[:n_train]
            self.mask = self.mask[:n_train]
            self.post_state = self.post_state[:n_train]
        elif split == "val":
            self.state = self.state[n_train:n_train+n_val]
            self.action = self.action[n_train:n_train+n_val]
            self.effect = self.effect[n_train:n_train+n_val]
            self.mask = self.mask[n_train:n_train+n_val]
            self.post_state = self.post_state[n_train:n_train+n_val]
        elif split == "test":
            self.state = self.state[n_train+n_val:]
            self.action = self.action[n_train+n_val:]
            self.effect = self.effect[n_train+n_val:]
            self.mask = self.mask[n_train+n_val:]
            self.post_state = self.post_state[n_train+n_val:]
        if n > 0:
            self.state = self.state[:n]
            self.action = self.action[:n]
            self.effect = self.effect[:n]
            self.mask = self.mask[:n]
            self.post_state = self.post_state[:n]

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        state = self.state[idx]
        a = self.action[idx]
        mask = self.mask[idx]
        state = torch.cat([state[:, :-1], self.binary[[state[:, -1].long()]]], dim=-1)
        if self.obj_relative:
            state[:, :3] = state[:, :3] - state[a[0], :3]  # just subtract the position

        post_state = self.post_state[idx]
        post_state = torch.cat([post_state[:, :-1], self.binary[[post_state[:, -1].long()]]], dim=-1)
        n_objects, _ = state.shape
        # [grasp_or_release, dx_loc, dy_loc, rot]
        action = torch.zeros(n_objects, 8, dtype=torch.float)
        action[a[0], :4] = torch.tensor([1, a[2], a[3], a[6]], dtype=torch.float)
        action[a[1], 4:] = torch.tensor([1, a[4], a[5], a[7]], dtype=torch.float)

        effect = torch.cat([self.effect[idx][:, :3], self.effect[idx][:, 9:12]], dim=-1)
        mask = torch.zeros(n_objects, dtype=torch.float)
        mask[:self.mask[idx]] = 1.0
        return state, action, effect, mask, post_state


def load_symbol_dataset(name, run, device):
    z_obj_pre = torch.load(wandb.restore(os.path.join(run.config["save_folder"], f"{name}_z_obj_pre.pt")).name).to(device)
    z_rel_pre = torch.load(wandb.restore(os.path.join(run.config["save_folder"], f"{name}_z_rel_pre.pt")).name).to(device)
    z_act = torch.load(wandb.restore(os.path.join(run.config["save_folder"], f"{name}_z_act.pt")).name).to(device)
    z_obj_post = torch.load(wandb.restore(os.path.join(run.config["save_folder"], f"{name}_z_obj_post.pt")).name).to(device)
    z_rel_post = torch.load(wandb.restore(os.path.join(run.config["save_folder"], f"{name}_z_rel_post.pt")).name).to(device)
    mask = torch.load(wandb.restore(os.path.join(run.config["save_folder"], f"{name}_mask.pt")).name).to(device)

    dataset = torch.utils.data.TensorDataset(z_obj_pre, z_rel_pre, z_act,
                                             z_obj_post, z_rel_post, mask)
    return dataset
