import pickle
import argparse
from copy import deepcopy

import torch
import lightning.pytorch as pl
import numpy as np

import dataset
import mcts
from learn_rules import collate_preds, Node, create_effect_classes
from models import load_ckpt

args = argparse.ArgumentParser()
args.add_argument("-n", type=str, required=True, help="Experiment name")
args = args.parse_args()

model, _ = load_ckpt(args.n, tag="best")
trainer = pl.Trainer(accelerator="cpu")
cnt_valloader = torch.utils.data.DataLoader(
    dataset.StateActionEffectDataset("blocks234", "val", obj_relative=False),
    batch_size=1024)
valset = collate_preds(trainer.predict(model, cnt_valloader))
valloader = torch.utils.data.DataLoader(valset, batch_size=1)
effect_to_class = torch.load(f"out/{args.n}/effect_to_class.pt")
val_effects, _, = create_effect_classes(valloader, given_effect_to_class=effect_to_class)

tree = pickle.load(open(f"out/{args.n}/tree.pkl", "rb"))
forward_fn = mcts.TreeForward(tree)

full_top5 = 0
acc_top5 = np.array([0, 0, 0, 0, 0])
full_acc = 0
acc = np.array([0, 0, 0, 0, 0])
miss_count = 0
for i, sample in enumerate(valloader):
    # z_i, r_i, a, z_f, r_f, m = sample
    # state = mcts.TreeSymbolicState((z_i[0], r_i[0]), (z_f[0], r_f[0]))
    all_accurate5 = True
    all_accurate = True
    missed = False
    for k in effect_to_class:
        node, bindings, path = forward_fn._classify(sample, tree[k])
        sorted_counts = sorted(node.counts.items(), key=lambda x: x[1], reverse=True)
        sorted_counts = [k for k, _ in sorted_counts]
        if val_effects[i][k] == -1:
            missed = True

        if val_effects[i][k] in sorted_counts[:1]:
            acc[k] += 1
        else:
            all_accurate = False

        if val_effects[i][k] in sorted_counts[:5]:
            acc_top5[k] += 1
        else:
            all_accurate5 = False

    if missed:
        miss_count += 1
    if all_accurate:
        full_acc += 1
    if all_accurate5:
        full_top5 += 1

    if (i+1) % 100 == 0:
        print(f"Acc: {acc / (i + 1)}, Top5: {acc_top5 / (i + 1)}")
        print(f"Acc: {full_acc / (i + 1):.4f}, Top5: {full_top5 / (i + 1):.4f}")
        print(f"Miss ratio: {miss_count / (i + 1):.4f}")
#     sorted_counts = sorted(node.counts.items(), key=lambda x: x[1], reverse=True)
#     sorted_counts = [k for k, _ in sorted_counts]
#     if val_effects[i] != -1:
#         if val_effects[i] in sorted_counts[:1]:
#             acc += 1
#         for _ in range(50):
#             next_state = forward_fn(state, action_vector=a[0])
#             if next_state.is_terminal():
#                 forw_acc += 1
#                 break
#         pred_count += 1
#     print(f"Acc: {acc / (pred_count):.4f}, Pred perc.: {pred_count / (i + 1):.4f}, Forw. acc: {forw_acc/ (pred_count)}",
#           end="\r")
# print(f"Acc: {acc / (pred_count):.8f}, Pred perc.: {pred_count / (i + 1):.8f}")
