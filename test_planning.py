import os
import sys

import torch
import numpy as np

from models import load_ckpt, DeepSym, MultiDeepSym, AttentiveDeepSym
import mcts


if __name__ == "__main__":
    cum_mean = []
    cum_std = []
    os_mean = []
    os_std = []
    for arg in sys.argv[1:]:
        print(f"Evaluating {arg}")
        model, _ = load_ckpt(arg, model_type=MultiDeepSym, tag="best")
        model.freeze()
        forward_fn = mcts.SubsymbolicForwardModel(model)

        cum_errors = {1: [], 2: [], 3: [], 4: [], 5: []}
        one_step_errors = {1: [], 2: [], 3: [], 4: [], 5: []}
        for a in range(1, 6):
            success = 0
            os_success = 0
            folder = os.path.join("out", "imgs2", str(a))
            for i in range(100):
                exp_folder = os.path.join(folder, str(i))
                states = torch.load(os.path.join(exp_folder, "states.pt"))
                actions = [line.strip() for line in open(os.path.join(exp_folder, "actions.txt")).readlines()[1:]]
                subsym_init = mcts.SubsymbolicState(states[0], states[-1])
                # root = mcts.MCTSNode(node_id=0, parent=None, state=subsym_init, forward_fn=forward_fn)
                # root.run(iter_limit=10000, time_limit=600, default_depth_limit=1, default_batch_size=1, n_proc=1)
                # _, plan, _, _ = root.plan(best_yield=True)
                # print(plan)
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
                    print(subsym_init.state[:, :3], file=open(os.path.join(exp_folder, "pred.txt"), "w"))
                    print(subsym_init.goal[:, :3], file=open(os.path.join(exp_folder, "pred.txt"), "a"))
                    print(subsym_init.state[:, :3] - subsym_init.goal[:, :3],
                          file=open(os.path.join(exp_folder, "pred.txt"), "a"))
                else:
                    print(subsym_init.state[:, :3], file=open(os.path.join(exp_folder, "pred.txt"), "w"))
                    print(subsym_init.goal[:, :3], file=open(os.path.join(exp_folder, "pred.txt"), "a"))
                    print(subsym_init.state[:, :3] - subsym_init.goal[:, :3],
                          file=open(os.path.join(exp_folder, "pred.txt"), "a"))

        cum_errors = {k: 100*torch.stack(v) for k, v in cum_errors.items()}
        one_step_errors = {k: 100*torch.stack(v) for k, v in one_step_errors.items()}

        cum_mean.append([cum_errors[a].mean(dim=0).tolist() for a in range(1, 6)])
        cum_std.append([cum_errors[a].std(dim=0).tolist() for a in range(1, 6)])
        os_mean.append([one_step_errors[a].mean(dim=0).tolist() for a in range(1, 6)])
        os_std.append([one_step_errors[a].std(dim=0).tolist() for a in range(1, 6)])

    cum_mean = np.array(cum_mean)
    cum_std = np.array(cum_std)
    os_mean = np.array(os_mean)
    os_std = np.array(os_std)
    np.save(os.path.join("out", "cum_mean.npy"), cum_mean)
    np.save(os.path.join("out", "cum_std.npy"), cum_std)
    np.save(os.path.join("out", "os_mean.npy"), os_mean)
    np.save(os.path.join("out", "os_std.npy"), os_std)
