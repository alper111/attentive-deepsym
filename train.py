"""Train DeepSym"""
import argparse

import torch
import yaml
import lightning.pytorch as pl

import utils
from dataset import StateActionEffectDataset

parser = argparse.ArgumentParser("Train DeepSym.")
parser.add_argument("-c", "--config", help="config file", type=str, required=True)
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

model = utils.create_model_from_config(config)

train_set = StateActionEffectDataset(config["dataset_name"], split="train")
val_set = StateActionEffectDataset(config["dataset_name"], split="val")
train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size"])

trainer = pl.Trainer(max_epochs=config["epoch"], gradient_clip_val=1.0)
trainer.fit(model, train_loader, val_loader)
