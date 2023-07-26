"""Train DeepSym"""
import argparse
import os

import yaml
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

from models import DeepSym, AttentiveDeepSym, MultiDeepSym, load_ckpt
from dataset import StateActionEffectDM

parser = argparse.ArgumentParser("Train DeepSym.")
parser.add_argument("-c", "--config", help="config file", type=str, required=True)
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

ckpt_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.join("logs", config["name"]),
                                             save_last=True, save_top_k=1, monitor="val_loss",
                                             mode="min")
logger = WandbLogger(name=config["name"], project="attentive-deepsym",
                     config=config, log_model=True, save_dir="logs", id=config["name"])
trainer = pl.Trainer(max_epochs=config["epoch"], gradient_clip_val=10.0,
                     logger=logger, devices=config["devices"], callbacks=[ckpt_callback])

ckpt_path = None
if config["resume"]:
    model_dict = {
        "attentive": AttentiveDeepSym,
        "multideepsym": MultiDeepSym,
        "deepsym": DeepSym
    }
    model, ckpt_path = load_ckpt(config["name"], model_type=model_dict[config["model"]], tag="latest")
else:
    if "model" not in config:
        model = AttentiveDeepSym(config)
    elif config["model"] == "attentive":
        model = AttentiveDeepSym(config)
    elif config["model"] == "multideepsym":
        model = MultiDeepSym(config)
    elif config["model"] == "deepsym":
        model = DeepSym(config)
    else:
        raise ValueError("Invalid model name.")

dm = StateActionEffectDM(config["dataset_name"], batch_size=config["batch_size"],
                         obj_relative=config["obj_relative"] if "obj_relative" in config else False)
trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
