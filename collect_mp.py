import time
import os
import argparse
import subprocess
import zipfile
import multiprocessing as mp

import torch
import wandb


def collect(script, num, t, folder, idx):
    subprocess.run(["python", script, "-N", num, "-T", t, "-o", folder, "-i", idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Collect interaction data in parallel.")
    parser.add_argument("-s", help="script", type=str, required=True)
    parser.add_argument("-d", help="data folder", type=str, required=True)
    parser.add_argument("-N", help="number of data per proc", type=int, required=True)
    parser.add_argument("-T", help="interaction per episode", type=int, required=True)
    parser.add_argument("-p", help="number of procs", type=int, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.d):
        os.makedirs(args.d)

    procs = []
    start = time.time()
    for i in range(args.p):
        p = mp.get_context("spawn").Process(target=collect,
                                            args=[args.s, str(args.N), str(args.T), args.d, str(i)])
        p.start()
        procs.append(p)

    for i in range(args.p):
        procs[i].join()
    end = time.time()
    elapsed = end - start
    print(args.N, file=open(os.path.join(args.d, "info.txt"), "w"))
    print(args.p, file=open(os.path.join(args.d, "info.txt"), "a"))
    print(f"Collected {args.p*args.N} samples in {elapsed:.2f} seconds. {args.p*args.N/elapsed}")
    print("Merging rolls...")
    keys = ["action", "effect", "mask", "state", "post_state"]
    for key in keys:
        field = []
        for i in range(args.p):
            field.append(torch.load(os.path.join(args.d, f"{key}_{i}.pt")))
        field = torch.cat(field, dim=0)
        torch.save(field, os.path.join(args.d, f"{key}.pt"))
        for i in range(args.p):
            os.remove(os.path.join(args.d, f"{key}_{i}.pt"))
    print("Done.")

    print("Uploading dataset to wandb as an artifact...")
    name = args.d.split("/")[-1]  # take the last part of the path
    with zipfile.ZipFile(f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(args.d):
            if file != ".DS_Store":
                zipf.write(os.path.join(args.d, file), arcname=file)
    wandb.init(project="attentive-deepsym", entity=wandb.api.default_entity, name=name+"_dataset")
    artifact = wandb.Artifact(name, type="dataset")
    artifact.add_file(f"{name}.zip")
    wandb.log_artifact(artifact)
    os.remove(f"{name}.zip")
