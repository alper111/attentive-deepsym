"""Train DeepSym"""
import argparse

import torch
import yaml
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

from models import AttentiveDeepSym
from dataset import StateActionEffectDataset

parser = argparse.ArgumentParser("Train DeepSym.")
parser.add_argument("-c", "--config", help="config file", type=str, required=True)
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

model = AttentiveDeepSym(config)

train_set = StateActionEffectDataset(config["dataset_name"], split="train")
val_set = StateActionEffectDataset(config["dataset_name"], split="val")
train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size"])

logger = WandbLogger(name=config["name"], project="attentive-deepsym",
                     config=config, log_model=True, save_dir="logs", id=config["name"])
trainer = pl.Trainer(max_epochs=config["epoch"], gradient_clip_val=1.0,
                     logger=logger, devices=config["devices"])
trainer.fit(model, train_loader, val_loader)
