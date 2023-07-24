import argparse
import os

import torch

from models import load_ckpt
import mcts

parser = argparse.ArgumentParser("Test planning")
parser.add_argument("-n", help="Experiment name", type=str)
args = parser.parse_args()

model, _ = load_ckpt(args.n, tag="best")
model.freeze()
forward_fn = mcts.SubsymbolicForwardModel(model)

cum_errors = {1: [], 2: [], 3: [], 4: [], 5: []}
one_step_errors = {1: [], 2: [], 3: [], 4: [], 5: []}
for a in range(1, 6):
    success = 0
    os_success = 0
    folder = os.path.join("out", "imgs3", str(a))
    for i in range(100):
        exp_folder = os.path.join(folder, str(i))
        states = torch.load(os.path.join(exp_folder, "states.pt"))
        actions = [line.strip() for line in open(os.path.join(exp_folder, "actions.txt")).readlines()[1:]]
        subsym_init = mcts.SubsymbolicState(states[0], states[-1])
        temp_st = mcts.SubsymbolicState(states[0], states[1])
        os_failed_yet = False
        for j, action in enumerate(actions):
            subsym_init = forward_fn(subsym_init, action, obj_relative=True)
            temp_st = forward_fn(temp_st, action, obj_relative=True)
            if j < len(actions) - 1:
                temp_st = mcts.SubsymbolicState(states[j+1], states[j+2])

        diff = (subsym_init.state[:, :3] - subsym_init.goal[:, :3]).abs().mean(dim=0)
        cum_errors[a].append(diff)
        os_diff = (temp_st.state[:, :3] - temp_st.goal[:, :3]).abs().mean(dim=0)
        one_step_errors[a].append(os_diff)

        if subsym_init.is_terminal():
            success += 1
        else:
            print(subsym_init.state[:, :3], file=open(os.path.join(exp_folder, "pred.txt"), "w"))
            print(subsym_init.goal[:, :3], file=open(os.path.join(exp_folder, "pred.txt"), "a"))
            print(subsym_init.state[:, :3] - subsym_init.goal[:, :3],
                  file=open(os.path.join(exp_folder, "pred.txt"), "a"))

for a in range(1, 6):
    print(f"Action {a} cum error: {torch.stack(cum_errors[a]).mean(dim=0)*100}")
    print(f"Action {a} one-step error: {torch.stack(one_step_errors[a]).mean(dim=0)*100}")
