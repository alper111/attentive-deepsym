import sys

import torch
import wandb
import numpy as np

import environment
import utils

text_items = {}
state_debug_texts = []
parameter_ids = {}
head_pre = 0
active_selector_pre = 0
obj_pre = -1
attention_arrow_ids = []
colors = [
    [0.7, 0.0, 0.0, 0.75],
    [0.0, 0.7, 0.0, 0.75],
    [0.0, 0.0, 0.7, 0.75],
    [0.7, 0.0, 0.7, 0.75],
    [0.7, 0.7, 0.0, 0.75],
]


def remove_attention_arrows(env):
    while len(attention_arrow_ids) > 0:
        arrow_id = attention_arrow_ids.pop()
        env._p.removeBody(arrow_id)


def draw_attention(env, attention, head, active_selector, obj):
    remove_attention_arrows(env)
    # object range
    min = 0
    max = env.num_objects
    global colors
    if obj != -1:
        min = obj
        max = obj+1

    attention = attention[head]
    for i in range(min, max):
        for k in range(0, env.num_objects):
            if attention[i, k] == active_selector:
                attention_arrow_ids.append(utils.create_arrow(env._p, env.get_obj_location(i), env.get_obj_location(k), color=colors[i % len(colors)]))


def draw_attention_loop(env, attention_maps):
    global head_pre, active_selector_pre, obj_pre
    print("===== ATTENTION VISUALIZER ======")
    print("head x : head x as active head")
    print("selec y : which element from attention will be visualized(0 or 1)")
    print("obj z : focus on obj z, -1 for all objects")
    print("act  : perform action")

    head = head_pre
    active_selector = active_selector_pre
    obj = obj_pre
    print(f"active head {head}")
    print(attention_maps[head])
    while True:
        com = input()
        com = com.split(" ")
        if com[0] == "head":
            head = int(com[1])
            print(attention_maps[head])
        elif com[0] == "selec":
            active_selector = int(com[1])
        elif com[0] == "obj":
            obj = int(com[1])
            obj = min(obj, env.num_objects-1)
        elif com[0] == "act":
            break
        if not (
            head == head_pre and
            active_selector == active_selector_pre and
            obj == obj_pre
        ):

            draw_attention(env, attention_maps, head, active_selector, obj)
            head_pre = head
            active_selector_pre = active_selector
            obj_pre = obj
    remove_attention_arrows(env)


def draw_single_symbols(env, z):
    global text_items
    global state_debug_texts
    for o_id in env.obj_dict:
        text_items[o_id] = env._p.addUserDebugText(str(o_id) + ":" + str(z[o_id].item()), [0., 0., 0.05],
                                                   textColorRGB=[1.0, 0.0, 0.0],
                                                   parentObjectUniqueId=env.obj_dict[o_id],
                                                   textSize=1)
        state_debug_texts.append(text_items[o_id])


def draw_action(env, z=None):

    from_obj = int(input("From: "))
    text_items[from_obj] = env._p.addUserDebugText(str(from_obj) + ":" + str(z[from_obj].item()), [0., 0., 0.05],
                                                   textColorRGB=[0.0, 0.0, 1.0],
                                                   parentObjectUniqueId=env.obj_dict[from_obj],
                                                   textSize=1.5,
                                                   replaceItemUniqueId=text_items[from_obj])
    state_debug_texts.append(text_items[from_obj])
    x_debug_texts = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            pos = [env.ds * i, env.ds * j, 0.01]
            x_debug_texts.append(env._p.addUserDebugText("*", pos,
                                                         textColorRGB=[0.0, 1.0, 1.0],
                                                         textSize=1.5,
                                                         parentObjectUniqueId=env.obj_dict[from_obj]))
    dxdy = input("From dxdy: ")
    dxdy = dxdy.split(" ")
    from_dx = int(dxdy[0])
    from_dy = int(dxdy[1])
    for debug_text in x_debug_texts:
        env._p.removeUserDebugItem(debug_text)
    x_debug_text = env._p.addUserDebugText("*", [env.ds * from_dx, env.ds * from_dy, 0.01],
                                           textColorRGB=[0.0, 1.0, 1.0],
                                           textSize=1.5,
                                           parentObjectUniqueId=env.obj_dict[from_obj])
    state_debug_texts.append(x_debug_text)
    to_obj = int(input("To: "))
    text_items[to_obj] = env._p.addUserDebugText(str(to_obj) + ":" + str(z[to_obj].item()), [0., 0., 0.05],
                                                 textColorRGB=[0.0, 0.0, 1.0],
                                                 parentObjectUniqueId=env.obj_dict[to_obj],
                                                 textSize=1.5,
                                                 replaceItemUniqueId=text_items[to_obj])
    state_debug_texts.append(text_items[to_obj])
    y_debug_texts = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            pos = [env.ds * i, env.ds * j, 0.01]
            y_debug_texts.append(env._p.addUserDebugText("*", pos,
                                                         textColorRGB=[0.0, 1.0, 1.0],
                                                         textSize=1.5,
                                                         parentObjectUniqueId=env.obj_dict[to_obj]))
    dxdy = input("To dxdy: ")
    dxdy = dxdy.split(" ")
    to_dx = int(dxdy[0])
    to_dy = int(dxdy[1])
    for debug_text in y_debug_texts:
        env._p.removeUserDebugItem(debug_text)
    y_debug_text = env._p.addUserDebugText("*", [env.ds * to_dx, env.ds * to_dy, 0.01],
                                           textColorRGB=[0.0, 1.0, 1.0],
                                           textSize=1.5,
                                           parentObjectUniqueId=env.obj_dict[to_obj])
    state_debug_texts.append(y_debug_text)

    init_rot, final_rot = input("Init rot, final rot: ").split(" ")
    init_rot = int(init_rot)
    final_rot = int(final_rot)
    action = [from_obj, to_obj, from_dx, from_dy, to_dx, to_dy, init_rot, final_rot]

    text_items[from_obj] = env._p.addUserDebugText(str(from_obj) + ":" + str(z[from_obj].item()), [0., 0., 0.05],
                                                   textColorRGB=[1.0, 0.0, 0.0],
                                                   parentObjectUniqueId=env.obj_dict[from_obj],
                                                   textSize=1.5,
                                                   replaceItemUniqueId=text_items[from_obj])
    state_debug_texts.append(text_items[from_obj])
    text_items[to_obj] = env._p.addUserDebugText(str(to_obj) + ":" + str(z[to_obj].item()), [0., 0., 0.05],
                                                 textColorRGB=[1.0, 0.0, 0.0],
                                                 parentObjectUniqueId=env.obj_dict[to_obj],
                                                 textSize=1.5,
                                                 replaceItemUniqueId=text_items[to_obj])
    state_debug_texts.append(text_items[to_obj])
    for debug_text in state_debug_texts:
        env._p.removeUserDebugItem(debug_text)

    return action, [x_debug_text, y_debug_text]


run_id = sys.argv[1]
run = wandb.init(entity="colorslab", project="multideepsym", resume="must", id=run_id)
config = (dict(run.config))
config["device"] = "cpu"
model = utils.create_model_from_config(config)
model.load("_best", from_wandb=True)
model.print_model()

for name in model.module_names:
    getattr(model, name).eval()


env = environment.BlocksWorld_v4(gui=1, min_objects=2, max_objects=2)
np.random.seed(int(sys.argv[2]))
env.reset()
parameter_ids["head"] = env._p.addUserDebugParameter("attention_heads", 0, config["n_attention_heads"]-1, 0)
parameter_ids["01"] = env._p.addUserDebugParameter("0/1 selector", 0, -1, 0)
parameter_ids["obj"] = env._p.addUserDebugParameter("object selector", -1, env.max_objects, -1)
parameter_ids["perform"] = env._p.addUserDebugParameter("perform action(click twice)", 0, -1, 0)
eff_arrow_ids = []

while True:
    one_hot = torch.tensor([[0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]])
    while len(eff_arrow_ids) > 0:
        arrow_id = eff_arrow_ids.pop()
        env._p.removeBody(arrow_id)
    state, types = env.state()
    state = torch.tensor(state)
    types = torch.tensor(types).reshape(-1, 1)
    state = torch.cat([torch.tensor(state), torch.tensor(types)], dim=-1)
    state = torch.cat([state[:, :-1], one_hot[[state[:, -1].long()]]], dim=-1)
    z = model.encode(state.unsqueeze(0), eval_mode=True)
    z = utils.binary_to_decimal(z[0, :, :].round())
    draw_single_symbols(env, z)
    att_weights = model.attn_weights(state.unsqueeze(0), torch.ones(1, state.shape[0]), eval_mode=True)
    draw_attention_loop(env, att_weights[0])
    action, debug_texts = draw_action(env, z)
    action_vector = torch.zeros(state.shape[0], 8, dtype=torch.float)
    action_vector[action[0], :4] = torch.tensor([1, action[2], action[3], action[6]], dtype=torch.float)
    action_vector[action[1], 4:] = torch.tensor([1, action[4], action[5], action[7]], dtype=torch.float)
    sample = {"state": state.unsqueeze(0),
              "action": action_vector.unsqueeze(0),
              "pad_mask": torch.ones(1, state.shape[0])}
    z, _, e_pred = model.forward(sample, eval_mode=True)
    print(e_pred[0, :, 2])
    e_pred = e_pred.detach()
    e_pred = e_pred.reshape(-1, 2, 3)
    for e_i, s_i in zip(e_pred, state):
        e_before, e_after = e_i
        from_before = s_i[:3].clone()
        from_after = s_i[:3].clone() + e_before[:3].clone()
        to_before = s_i[:3].clone()
        to_before[2] = 0.9
        to_after = to_before.clone() + e_after[:3].clone()
        eff_arrow_ids.append(utils.create_arrow(env._p, from_before, from_after, color=[1.0, 0.0, 0.0, 0.75]))
        eff_arrow_ids.append(utils.create_arrow(env._p, to_before, to_after, color=[0.0, 1.0, 0.0, 0.75]))
    env.step(*action)
    for debug_text in debug_texts:
        env._p.removeUserDebugItem(debug_text)
