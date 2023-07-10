import time
import os
import argparse

import torch
import numpy as np

import environment
import utils


def collect_rollout(env):
    _, depth_a, seg_a = env.state()
    depth_a = utils.normalize_depth_img(depth_a)
    action = env.sample_random_action()
    effect = env.step(*action)
    _, depth_b, seg_b = env.state()
    return (depth_a, seg_a), (depth_b, seg_b), action, effect


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Explore environment.")
    parser.add_argument("-N", help="number of interactions", type=int, required=True)
    parser.add_argument("-o", help="output folder", type=str, required=True)
    parser.add_argument("-i", help="offset index", type=int, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.o):
        os.makedirs(args.o)

    env = environment.BlocksWorld_v2(gui=0, min_objects=1, max_objects=3)
    # env = environment.BlocksWorld_v3(gui=0, min_objects=5, max_objects=9)
    np.random.seed()

    states = torch.zeros(args.N, 1, 256, 256, dtype=torch.uint8)
    segmentations = torch.zeros(args.N, 256, 256, dtype=torch.uint8)
    actions = torch.zeros(args.N, 2, dtype=torch.int32)
    # actions = torch.zeros(args.N, 4, dtype=torch.int32)
    effects = torch.zeros(args.N, env.max_objects, 7, dtype=torch.float)

    post_states = torch.zeros(args.N, 1, 256, 256, dtype=torch.uint8)
    post_segmentations = torch.zeros(args.N, 256, 256, dtype=torch.uint8)

    prog_it = args.N // 20
    start = time.time()
    env_it = 0
    i = 0
    while i < args.N:
        env_it += 1
        (depth_a, seg_a), (depth_b, seg_b), (from_idx, to_idx), effect = collect_rollout(env)
        if seg_a.max() < 4:
            env_it = 0
            env.reset_objects()
            continue

        states[i, 0] = torch.tensor(depth_a, dtype=torch.uint8)
        segmentations[i] = torch.tensor(seg_a, dtype=torch.uint8)
        post_states[i, 0] = torch.tensor(depth_b, dtype=torch.uint8)
        post_segmentations[i] = torch.tensor(seg_b, dtype=torch.uint8)
        actions[i, 0], actions[i, 1] = from_idx, to_idx
        # actions[i, 0], actions[i, 1], actions[i, 2], actions[i, 3] = from_idx[0], from_idx[1], to_idx[0], to_idx[1]
        # effects[i, :env.num_objects] = torch.tensor(effect, dtype=torch.float)
        if env.num_objects == 1 and env_it == 4:
            env_it = 0
            env.reset_objects()
        elif env.num_objects == 2 and env_it == 8:
            env_it = 0
            env.reset_objects()
        elif (env_it) == 20:
            env_it = 0
            env.reset_objects()
        # if (env_it) == 20:
        #     env_it = 0
        #     env.reset_objects()

        i += 1
        if i % prog_it == 0:
            print(f"Proc {args.i}: {100*i/args.N}% completed.")

    torch.save(states, os.path.join(args.o, f"state_{args.i}.pt"))
    torch.save(post_states, os.path.join(args.o, f"post_state_{args.i}.pt"))
    torch.save(actions, os.path.join(args.o, f"action_{args.i}.pt"))
    torch.save(effects, os.path.join(args.o, f"effect_{args.i}.pt"))
    torch.save(segmentations, os.path.join(args.o, f"segmentation_{args.i}.pt"))
    torch.save(post_segmentations, os.path.join(args.o, f"post_segmentation_{args.i}.pt"))
    end = time.time()
    del env
    print(f"Completed in {end-start:.2f} seconds.")
