import sys
import os

import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib

from models import load_ckpt, AttentiveDeepSym, MultiDeepSym, DeepSym
from dataset import StateActionEffectDataset
import utils


def compute_symbols(model, loader):
    symbols = {}
    for batch in loader:
        state, _, _, _, _ = batch
        if type(model) == DeepSym:
            st, _, _, _, _ = model._preprocess_batch(batch)
            z = model.encode(st, eval_mode=True)
            z_str = utils.binary_tensor_to_str(z)
            for z_j, s_j in zip(z_str, st):
                if z_j not in symbols:
                    symbols[z_j] = []
                symbols[z_j].append(s_j)
        else:
            z = model.encode(state, eval_mode=True)
            for z_i, s_i in zip(z, state):
                z_str = utils.binary_tensor_to_str(z_i)
                for z_j, s_j in zip(z_str, s_i):
                    if z_j not in symbols:
                        symbols[z_j] = []
                    symbols[z_j].append(s_j)
    for k, v in symbols.items():
        symbols[k] = torch.stack(v)
    return symbols


def compute_relations(model, loader):
    relations = {}
    for batch in loader:
        state, _, _, mask, _ = batch
        r = model.attn_weights(state, mask, eval_mode=True)
        for r_i, s_i in zip(r, state):
            for j, r_ij in enumerate(r_i):
                if j not in relations:
                    relations[j] = {0: {"q": [], "k": []}, 1: {"q": [], "k": []}}
                for o1, r_ijk in enumerate(r_ij):
                    for o2, r_ijkl in enumerate(r_ijk):
                        relations[j][r_ijkl.item()]["q"].append(s_i[o1])
                        relations[j][r_ijkl.item()]["k"].append(s_i[o2])
    for k, v in relations.items():
        for k2, v2 in v.items():
            relations[k][k2]["q"] = torch.stack(v2["q"])
            relations[k][k2]["k"] = torch.stack(v2["k"])
    return relations


def plot_symbols(sym_dict, sym_per_row, out_path):
    symbols = {k: v for k, v in sym_dict.items() if len(v) > 100}
    unique_symbols = list(symbols.keys())
    n_rows = len(unique_symbols) // sym_per_row + 1
    n_rows -= 1 if len(unique_symbols) % sym_per_row == 0 else 0
    fig, ax = plt.subplots(n_rows, sym_per_row, figsize=(sym_per_row * 4, n_rows * 4))
    cmap = plt.get_cmap("coolwarm")
    matplotlib.rc('font', size=16)
    for i, z_i in enumerate(unique_symbols):
        row_num = i // sym_per_row
        col_num = i % sym_per_row
        curr_ax = ax[row_num, col_num] if n_rows > 1 else ax[col_num]
        alpha_per_data = 0.5
        normalized_depths = (symbols[z_i][:, 2] + 0.05) / (0.1)
        obj_types = torch.argmax(symbols[z_i][:, -5:], dim=1)[:1000]
        mask, = torch.where(obj_types == 1)
        if len(mask) > 0:
            colors = [cmap(d) for j, d in enumerate(normalized_depths[:1000]) if obj_types[j] == 1]
            curr_ax.scatter(symbols[z_i][mask, 1], symbols[z_i][mask, 0], alpha=alpha_per_data, c=colors,
                            marker="o", linewidths=0.2, edgecolors="black")
        mask, = torch.where(obj_types == 4)
        if len(mask) > 0:
            colors = [cmap(d) for j, d in enumerate(normalized_depths[:1000]) if obj_types[j] == 4]
            curr_ax.scatter(symbols[z_i][mask, 1], symbols[z_i][mask, 0], alpha=alpha_per_data, c=colors,
                            marker="^", linewidths=0.2, edgecolors="black")
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=curr_ax, ticks=[0.1, 0.5, 0.9],
                            fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(["-0.04", "0", "0.04"])
        curr_ax.set_title(z_i + f" ({len(symbols[z_i]):,})")
        curr_ax.set_xlim([-0.5, 0.5])
        curr_ax.set_ylim([-0.5, 0.5])
        # set ticks
        curr_ax.set_xticks([-0.4, 0, 0.4])
        curr_ax.set_yticks([-0.4, 0, 0.4])
        # set tick labels
        curr_ax.set_xticklabels(["-0.4", "0", "0.4"], fontsize=16)
        curr_ax.set_yticklabels(["-0.4", "0", "0.4"], fontsize=16)
        # make it square
        curr_ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    pdf = PdfPages(os.path.join(out_path, "symbols.pdf"))
    pdf.savefig(fig)
    pdf.close()


def plot_relations(rel_dict, rel_per_row, out_path):
    # only use rel=1 for now
    rel_dict = {k: v[1] for k, v in rel_dict.items() if len(v[1]["q"]) > 100}
    unique_relations = list(rel_dict.keys())
    n_rows = len(unique_relations) // rel_per_row + 1
    n_rows -= 1 if len(unique_relations) % rel_per_row == 0 else 0
    fig, ax = plt.subplots(n_rows, rel_per_row, figsize=(rel_per_row * 4, n_rows * 4))
    cmap = plt.get_cmap("coolwarm")
    matplotlib.rc('font', size=16)
    for i, r_i in enumerate(unique_relations):
        row_num = i // rel_per_row
        col_num = i % rel_per_row
        curr_ax = ax[row_num, col_num] if n_rows > 1 else ax[col_num]
        alpha_per_data = 0.5
        deltas = rel_dict[r_i]["q"][:, :3] - rel_dict[r_i]["k"][:, :3]
        normalized_depths = (deltas[:, 2] + 0.05) / (0.1)
        colors = [cmap(d) for d in normalized_depths[:1000]]
        curr_ax.scatter(deltas[:1000, 1], deltas[:1000, 0], alpha=alpha_per_data, c=colors,
                        marker="o", linewidths=0.2, edgecolors="black")
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=curr_ax, ticks=[0.1, 0.5, 0.9],
                            fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(["-0.04", "0", "0.04"])
        curr_ax.set_xlim([-0.5, 0.5])
        curr_ax.set_ylim([-0.5, 0.5])
        # set ticks
        curr_ax.set_xticks([-0.4, 0, 0.4])
        curr_ax.set_yticks([-0.4, 0, 0.4])
        # set tick labels
        curr_ax.set_xticklabels(["-0.4", "0", "0.4"], fontsize=16)
        curr_ax.set_yticklabels(["-0.4", "0", "0.4"], fontsize=16)
        # make it square
        curr_ax.set_aspect('equal', adjustable='box')
        curr_ax.set_title(f"R{r_i}: {len(rel_dict[r_i]['q']):,}")

    plt.tight_layout()
    pdf = PdfPages(os.path.join(out_path, "relations.pdf"))
    pdf.savefig(fig)
    pdf.close()


if __name__ == "__main__":
    models = {"deepsym": DeepSym, "multideepsym": MultiDeepSym, "attentive": AttentiveDeepSym}
    model_type = models[sys.argv[1]]
    run_id = sys.argv[2]
    model, _ = load_ckpt(run_id, model_type=model_type, tag="best")
    model.freeze()

    trainset = StateActionEffectDataset("blocks2_v2", split="train", obj_relative=True)
    loader = torch.utils.data.DataLoader(trainset, batch_size=1024)

    out_path = os.path.join("out", run_id)
    sym_path = os.path.join(out_path, "symbols.pt")
    rel_path = os.path.join(out_path, "relations.pt")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f"Computing symbols for {run_id}...")
        symbols = compute_symbols(model, loader)
        torch.save(symbols, sym_path)
        if type(model) == AttentiveDeepSym:
            print(f"Computing relations for {run_id}...")
            relations = compute_relations(model, loader)
            torch.save(relations, rel_path)
    else:
        symbols = torch.load(sym_path)
        if type(model) == AttentiveDeepSym:
            relations = torch.load(rel_path)

    plot_symbols(symbols, 4, out_path)
    plot_relations(relations, 4, out_path)
