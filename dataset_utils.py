import torch
import numpy as np
from dataset import StateActionEffectDataset
import argparse
import os
import zipfile

import wandb

def metrics_by_name(dataset_name):
    train = StateActionEffectDataset(dataset_name, "train")
    val = StateActionEffectDataset(dataset_name, "val")
    total = StateActionEffectDataset(dataset_name, "")
    path = os.path.join("data", dataset_name)
    mtext = ("====TRAIN SET=====\n")
    mtext += metrics(train)
    mtext +=("====VAL SET=====\n")
    mtext +=metrics(val)
    mtext +=("====TOTAL=====\n")
    mtext +=metrics(total)
    f = open(os.path.join(path, "metrics.txt"), "w+")
    f.write(mtext)
    f.close()
def metrics(dataset):
    metrics = ""
    metrics += (f"{len(dataset)} samples \n")
    count = 0
    for e in dataset.effect:
        if torch.any(e[:,2].abs().max() > 0.1):
            count += 1
    metrics += (f"{count / len(dataset) * 100}% resulted in ds\n")
    metrics += (f"{count } samples\n")
    count = 0
    for e in dataset.effect:
        if np.count_nonzero(e[:,2] > 0.1) == 2:
            count += 1
    metrics +=(f"{count / len(dataset) * 100:.2f}% multi object(2) \n")
    metrics += (f"{count } samples\n")
    count = 0
    for e in dataset.effect:
        if np.count_nonzero(e[:,2] > 0.1) > 2:
            count += 1
    metrics +=(f"{count / len(dataset) * 100:.2f}% multi object(3) movement\n")
    metrics +=(f"{count } samples\n")
    count = 0
    for e in dataset.effect:
        if np.count_nonzero(e[:,2] < -0.1) > 0 or np.count_nonzero(e[:,2 + 9] < -0.1) > 1:
            count += 1
    metrics += (f"{count / len(dataset) * 100:.2f}% towers falling down\n")
    metrics +=(f"{count } samples\n")
    count = 0
    for sample in dataset:
        target = torch.argmax(sample["action"][:,4])
        if sample["effect"][target][2] < 0.1:
            if torch.any((((sample["action"][:,0]) != -1).view((-1,1)) * sample["effect"])[:,2] > 0.2):
                count += 1
    metrics += (f"{count / len(dataset) * 100:.2f}% mistargets\n")
    metrics +=(f"{count } samples\n")
    num_objects = {}
    for mask in dataset.mask:
        mask = mask.item()
        if mask  in num_objects.keys():
            num_objects[mask] += 1
        else:
            num_objects[mask] = 1
    metrics += str(num_objects)
    return metrics
def merge_datasets(args):

    path = os.path.join("data", args.large)
    l_state = torch.load(os.path.join(path, "state.pt"))
    l_action = torch.load(os.path.join(path, "action.pt"))
    l_effect = torch.load(os.path.join(path, "effect.pt"))
    l_mask = torch.load(os.path.join(path, "mask.pt"))
    l_post_state = torch.load(os.path.join(path, "post_state.pt"))


    path = os.path.join("data", args.small)
    s_state = torch.load(os.path.join(path, "state.pt"))
    s_action = torch.load(os.path.join(path, "action.pt"))
    s_effect = torch.load(os.path.join(path, "effect.pt"))
    s_mask = torch.load(os.path.join(path, "mask.pt"))
    s_post_state = torch.load(os.path.join(path, "post_state.pt"))

    len_small = s_state.shape[0]
    len_large = l_state.shape[0]
    while s_state.shape[1] != l_state.shape[1]:
        if s_state.shape[1] < l_state.shape[1]:
            s_state = torch.cat((s_state, torch.zeros((s_state.shape[0],1 ,s_state.shape[2]))), dim = 1)
            s_post_state = torch.cat((s_post_state, torch.zeros((s_state.shape[0],1 ,s_state.shape[2]))), dim = 1)
            s_effect = torch.cat((s_effect, torch.zeros((s_effect.shape[0],1 ,s_effect.shape[2]))), dim = 1)
        else:
            l_state = torch.cat((l_state, torch.zeros((l_state.shape[0],1 ,l_state.shape[2]))), dim = 1)
            l_post_state = torch.cat((l_post_state, torch.zeros((l_state.shape[0],1 ,l_state.shape[2]))), dim = 1)
            l_effect = torch.cat((l_effect, torch.zeros((l_effect.shape[0],1 ,l_effect.shape[2]))), dim = 1)
    divide_index = int(len_small/2)
    action = torch.concat([s_action, l_action] , dim=0)
    state = torch.concat([s_state, l_state], dim=0)
    effect = torch.concat([s_effect, l_effect], dim=0)
    mask = torch.concat([s_mask, l_mask], dim=0)
    post_state = torch.concat([s_post_state, l_post_state], dim=0)

    shuffle = torch.randperm(action.size()[0])

    action=action[shuffle]
    state=state[shuffle]
    effect=effect[shuffle]
    mask=mask[shuffle]
    post_state=post_state[shuffle]
    args.o = os.path.join("./data", args.o)
    if not os.path.exists(args.o):
            os.makedirs(args.o)

    torch.save(state, os.path.join(args.o, f"state.pt"))
    torch.save(action, os.path.join(args.o, f"action.pt"))
    torch.save(mask, os.path.join(args.o, f"mask.pt"))
    torch.save(effect, os.path.join(args.o, f"effect.pt"))
    torch.save(post_state, os.path.join(args.o, f"post_state.pt"))

def merge_rolls(args):
    keys = ["action", "effect", "mask", "state", "post_state"]

    output_folder = os.path.join("./data", args.o)
    for key in keys:
        field = torch.cat([torch.load(os.path.join(output_folder, f"{key}_{i}.pt")) for i in range(args.i)], dim=0)
        torch.save(field, os.path.join(os.path.join(output_folder, f"{key}.pt")))
        for i in range(args.i):
            os.remove(os.path.join(output_folder, f"{key}_{i}.pt"))
    metrics_by_name(args.o)


def upload_dataset_to_wandb(name, path):
    with zipfile.ZipFile(f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(path):
            if file != ".DS_Store":
                zipf.write(os.path.join(path, file), arcname=file)
    wandb.init(project="multideepsym", entity="colorslab")
    artifact = wandb.Artifact(name, type="dataset")
    artifact.add_file(f"{name}.zip")
    wandb.log_artifact(artifact)
    os.remove(f"{name}.zip")


def get_dataset_from_wandb(name):
    artifact = wandb.use_artifact(f"colorslab/multideepsym/{name}:latest", type="dataset")
    artifact_dir = artifact.download()
    archive = zipfile.ZipFile(os.path.join(artifact_dir, f"{name}.zip"), "r")
    archive.extractall(os.path.join("data", name))
    archive.close()
    os.remove(os.path.join(artifact_dir, f"{name}.zip"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser("See dataset metrics.")
    parser.add_argument("action", type=str)
    parser.add_argument("-o", help="dataset name", type=str)
    parser.add_argument("-s", "--small", help="smaller dataset", type=str)
    parser.add_argument("-l", "--large", help="appended dataset", type=str)
    parser.add_argument("-i", help="number of rolls", type=int)

    args = parser.parse_args()
    if args.action == "metrics":
        metrics_by_name(args.o)
    if args.action == "merge_datasets":
        merge_datasets(args)
    if args.action == "merge_rolls":
        merge_rolls(args)
    if args.action == "upload":
        name = args.o
        metrics_by_name(name)
        path = os.path.join("./data", name)
        upload_dataset_to_wandb(name, path)
    if args.action == "download":
        wandb.init(project="multideepsym", entity="colorslab")
        get_dataset_from_wandb(args.o)
