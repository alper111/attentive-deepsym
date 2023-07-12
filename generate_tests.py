import os
from PIL import Image

import torch
import numpy as np

import environment


def random_action(num_objects):
    obj1 = np.random.randint(0, num_objects)
    dx1 = np.random.randint(-1, 2)
    obj2 = np.random.randint(0, num_objects)
    dx2 = np.random.randint(-1, 2)
    return f"{obj1},0,{dx1},{obj2},0,{dx2}"


one_hot = torch.tensor([[0, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0]])
env = environment.BlocksWorld_v4(gui=0, min_objects=2, max_objects=2)

img_out_path = "out/imgs"
if not os.path.exists(img_out_path):
    os.makedirs(img_out_path)

for n_action in range(1, 6):
    subsymbolic_success = 0
    symbolic_success = 0
    dict_success = 0
    dict_found = 0
    i = 0
    seed_num = 0
    while i < 100:
        seed_num += 1
        skip = False
        current_folder = os.path.join(img_out_path, str(n_action), str(i))
        if not os.path.exists(current_folder):
            os.makedirs(current_folder)

        np.random.seed(seed_num)
        env.reset()

        actions = []
        for _ in range(n_action):
            a = env.full_random_action()
            actions.append(f"{a[0]},0,{a[3]},{a[1]},0,{a[5]}")

        states = []
        # initial state
        poses, types = env.state()
        state = torch.tensor(np.hstack([poses, types.reshape(-1, 1)]))
        init_state = torch.cat([state[:, :-1], one_hot[[state[:, -1].long()]]], dim=-1)
        states.append(init_state)
        # torch.save(init_state, os.path.join(current_folder, "init_state.pt"))

        # act
        print("Actions:", file=open(os.path.join(current_folder, "actions.txt"), "w"))
        all_imgs = []
        for j, action in enumerate(actions):
            print(action, file=open(os.path.join(current_folder, "actions.txt"), "a"))
            action = [int(x) for x in action.split(",")]
            _, effect, _, images = env.step(action[0], action[3], action[1], action[2], action[4], action[5],
                                            rotated_grasp=1, rotated_release=1, get_images=True)
            poses, types = env.state()
            state = torch.tensor(np.hstack([poses, types.reshape(-1, 1)]))
            state = torch.cat([state[:, :-1], one_hot[[state[:, -1].long()]]], dim=-1)
            states.append(state)
            if effect[action[0], 2] < 0.1:
                skip = True
                break
            all_imgs.append(images)

        for j, images in enumerate(all_imgs):
            for k, img in enumerate(images):
                Image.fromarray(img).save(os.path.join(current_folder, f"{j}_{k}.png"))

        if skip:
            os.system(f"rm -rf {current_folder}")
            continue
        print(seed_num, file=open(os.path.join(current_folder, "seed.txt"), "w"))

        # final state
        torch.save(torch.stack(states), os.path.join(current_folder, "states.pt"))
        i += 1
        print(f"n_action={n_action}, i={i}, seed={seed_num}")
