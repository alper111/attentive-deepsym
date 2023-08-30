import argparse

import numpy as np
import torch
import torchvision

from models import load_ckpt
import environment
import utils
import mcts


def save_image(env, name):
    rgb, _, _ = utils.get_image(env._p, eye_position=[1.9, 0.0, 1.4],
                                target_position=[0.9, 0.0, 0.5], up_vector=[0, 0, 1],
                                width=512, height=512)
    torchvision.utils.save_image(torch.tensor(rgb / 255.0, dtype=torch.float).permute(2, 0, 1)[:3], name)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-n", type=str, required=True, help="Experiment name")
    args.add_argument("-s", type=int, default=0, help="Random seed")
    args = args.parse_args()

    model, _ = load_ckpt(args.n, tag="best")
    model.freeze()
    forward_fn = mcts.SubsymbolicForwardModel(model, obj_relative=True)
    env = environment.BlocksWorld_v4(gui=1, min_objects=4, max_objects=4)
    np.random.seed(int(args.s))
    env.reset_objects()

    it = 0
    seed = int(args.s)
    prev_z = None
    prev_r = None
    while True:
        it += 1
        state = torch.tensor(env.state_concat(), dtype=torch.float).unsqueeze(0)
        n_obj = state.shape[1]
        pad_mask = torch.ones(1, n_obj)
        z = model.encode(state, eval_mode=True)
        r = model.attn_weights(state, pad_mask, eval_mode=True)
        if prev_z is not None:
            _, obj_diff, _ = torch.where(z != prev_z)
            _, rel_diff, o1_diff, o2_diff = torch.where(r != prev_r)
            rel_diff = torch.stack([rel_diff, o1_diff, o2_diff], dim=-1)
            obj_diff = obj_diff.unique()
        print(" " * 10, end="")
        print("Obj. Symbol", end="")
        print(" " * 9, end="")
        print("Rel. 1", end="")
        print(" " * 11, end="")
        print("Rel. 2", end="")
        print(" " * 11, end="")
        print("Rel. 3", end="")
        print(" " * 11, end="")
        print("Rel. 4")
        print(" " * 28, end="")
        print("0  1  2  3", end="")
        print(" " * 7, end="")
        print("0  1  2  3", end="")
        print(" " * 7, end="")
        print("0  1  2  3", end="")
        print(" " * 7, end="")
        print("0  1  2  3")
        for i in range(n_obj):
            obj_str = f"{z[0, i, :].int().tolist()}"
            if (prev_z is not None) and (i in obj_diff):
                obj_str = f"\033[91m{obj_str}\033[0m"
            print_str = f"Object {i}: {obj_str}"
            print(print_str, end="")
            for k in range(4):
                print_str = f" -- {i}["
                for j in range(4):
                    rel_str = f"{r[0, k, i, j].int()}"
                    if (prev_r is not None) and ((torch.tensor([k, i, j]) == rel_diff).all(dim=-1).any().item()):
                        rel_str = f"\033[91m{rel_str}\033[0m"
                    print_str += f"{rel_str}"
                    if j < 3:
                        print_str += ", "
                print_str += "]"
                print(print_str, end="")
            print()

        prev_z = z.clone()
        prev_r = r.clone()

        # act
        action_str = input("Action: ")
        if action_str[:5] == "reset":
            _, seed = action_str.split(" ")
            seed = int(seed)
            it = 0
            np.random.seed(int(seed))
            env.reset_objects()
            prev_z = None
            prev_r = None
            continue

        action = action_str.split(",")
        action = [int(a) for a in action]
        action_str = f"{action[0]},0,{action[1]},{action[2]},0,{action[3]}"
        init_st = mcts.SubsymbolicState(state[0], state[0])
        next_st = forward_fn(init_st, action_str)
        arrows = []
        for obj_before, obj_after in zip(init_st.state, next_st.state):
            arrows.append(utils.create_arrow(env._p, obj_before[:3], obj_after[:3]))
        save_image(env, f"{seed}_{it}.png")
        env.step(action[0], action[2], 0, action[1], 0, action[3], 1, 1)
        for arrow in arrows:
            env._p.removeBody(arrow)
