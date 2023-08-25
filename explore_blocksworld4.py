import time
import os
import argparse

import torch
import numpy as np
import environment


buffer = []


def collect_rollout(env):
    action = env.full_random_action()
    pre_position, effect = env.step(*action)
    post_position = env.state()
    return pre_position, action, effect, post_position


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Explore environment.")
    parser.add_argument("-N", help="number of interactions", type=int, required=True)
    parser.add_argument("-T", help="interaction per episode", type=int, required=True)
    parser.add_argument("-o", help="output folder", type=str, required=True)
    parser.add_argument("-i", help="offset index", type=int, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.o):
        os.makedirs(args.o)
    env = environment.BlocksWorld_v4(gui=1, min_objects=2, max_objects=4)
    np.random.seed()

    # (pos, quat, type)
    states = torch.zeros(args.N, env.max_objects, 11, dtype=torch.float)
    # (obj_i, obj_j, from_x, from_y, to_x, to_y, rot_init, rot_final)
    actions = torch.zeros(args.N, 8, dtype=torch.int)
    # how many objects are there in the scene
    masks = torch.zeros(args.N, dtype=torch.int)
    # (∆pos_grasp, ∆quat_grasp, ∆pos_release, ∆quat_release)
    effects = torch.zeros(args.N, env.max_objects, 14, dtype=torch.float)
    post_states = torch.zeros(args.N, env.max_objects, 11, dtype=torch.float)

    start = time.time()
    env_it = 0
    i = 0

    while i < args.N:
        if (env_it) == args.T:
            env_it = 0
            env.reset_objects()

        pre_position, action, effect, post_position = collect_rollout(env)
        # if the target object couldn't be picked up, skip with a probability
        if effect[action[0], 2] < 0.1 and np.random.rand() < 0.8:
            env_it += 1
            continue
        env_it += 1
        states[i, :env.num_objects] = torch.tensor(pre_position, dtype=torch.float)
        actions[i] = torch.tensor(action, dtype=torch.int)
        masks[i] = env.num_objects
        effects[i, :env.num_objects] = torch.tensor(effect, dtype=torch.float)
        post_states[i, :env.num_objects] = torch.tensor(post_position, dtype=torch.float)

        i += 1
        if i % (args.N // 100) == 0:
            print(f"Proc {args.i}: {100*i/args.N}% completed.")

    torch.save(states, os.path.join(args.o, f"state_{args.i}.pt"))
    torch.save(actions, os.path.join(args.o, f"action_{args.i}.pt"))
    torch.save(masks, os.path.join(args.o, f"mask_{args.i}.pt"))
    torch.save(effects, os.path.join(args.o, f"effect_{args.i}.pt"))
    torch.save(post_states, os.path.join(args.o, f"post_state_{args.i}.pt"))
    end = time.time()
    del env
    print(f"Completed in {end-start:.2f} seconds.")
