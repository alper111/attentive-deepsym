import os
import zipfile
from itertools import permutations

import wandb
import yaml
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torchvision import transforms

import blocks
import models


def parse_and_init(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    # init run
    run = wandb.init(project="multideepsym", entity="colorslab", config=config)
    # create a save folder if not exists
    save_folder = run.config["save_folder"]
    os.makedirs(save_folder, exist_ok=True)
    # also save the config file in the save folder
    with open(os.path.join(save_folder, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # download and extract dataset if not exists
    data_path = os.path.join("data", config["dataset_name"])
    if not os.path.exists(data_path):
        get_dataset_from_wandb(config["dataset_name"])

    return wandb.config


def create_model_from_config(config):
    # create the encoder
    enc_layers = [config["state_dim"]] + \
                 [config["hidden_dim"]]*config["n_hidden_layers"] + \
                 [config["latent_dim"]]
    enc_mlp = blocks.MLP(enc_layers, batch_norm=config["batch_norm"])
    encoder = torch.nn.Sequential(
        enc_mlp,
        blocks.GumbelSigmoidLayer(hard=config["gumbel_hard"],
                                  T=config["gumbel_t"])
    )
    pre_att_enc_layers = [config["state_dim"]] + \
                         [config["hidden_dim"]]*config["n_hidden_layers"]
    pre_att_enc = blocks.MLP(pre_att_enc_layers, batch_norm=config["batch_norm"])
    attention = blocks.GumbelAttention(in_dim=config["hidden_dim"],
                                       out_dim=config["hidden_dim"],
                                       num_heads=config["n_attention_heads"])

    # create a feedforward net to process input before attention
    post_enc_layers = [config["latent_dim"]+config["action_dim"]] + \
                      [config["hidden_dim"]]*config["n_hidden_layers"]
    post_enc = blocks.MLP(post_enc_layers, batch_norm=config["batch_norm"])

    # create the decoder
    dec_layers = [config["hidden_dim"]*config["n_attention_heads"]] + \
                 [config["hidden_dim"]]*(config["n_hidden_layers"]) + \
                 [config["effect_dim"]]
    decoder = blocks.MLP(dec_layers, batch_norm=config["batch_norm"])

    # create the model
    model = models.AttentiveDeepSym(encoder=encoder, decoder=decoder, attention=attention,
                                    post_encoder=post_enc, pre_attention=pre_att_enc,
                                    lr=config["lr"], loss_coeff=config["loss_coeff"])

    return model


def connect(gui=1):
    import pkgutil
    from pybullet_utils import bullet_client
    if gui:
        p = bullet_client.BulletClient(connection_mode=bullet_client.pybullet.GUI)
    else:
        p = bullet_client.BulletClient(connection_mode=bullet_client.pybullet.DIRECT)
        egl = pkgutil.get_loader("eglRenderer")
        if (egl):
            p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        else:
            p.loadPlugin("eglRendererPlugin")
    return p


def create_object(p, obj_type, size, position, rotation=[0, 0, 0], mass=1, color=None, with_link=False):
    collisionId = -1
    visualId = -1

    if obj_type == p.GEOM_SPHERE:
        collisionId = p.createCollisionShape(shapeType=obj_type, radius=size[0])

        if color == "random":
            color = np.random.rand(3).tolist() + [1]
            visualId = p.createVisualShape(shapeType=obj_type, radius=size[0], rgbaColor=color)
        elif color is not None:
            visualId = p.createVisualShape(shapeType=obj_type, radius=size[0], rgbaColor=color)

    elif obj_type in [p.GEOM_CAPSULE, p.GEOM_CYLINDER]:
        collisionId = p.createCollisionShape(shapeType=obj_type, radius=size[0], height=size[1])

        if color == "random":
            color = np.random.rand(3).tolist() + [1]
            visualId = p.createVisualShape(shapeType=obj_type, radius=size[0], length=size[1], rgbaColor=color)
        elif color is not None:
            visualId = p.createVisualShape(shapeType=obj_type, radius=size[0], length=size[1], rgbaColor=color)

    elif obj_type == p.GEOM_BOX:
        collisionId = p.createCollisionShape(shapeType=obj_type, halfExtents=size)

        if color == "random":
            color = np.random.rand(3).tolist() + [1]
            visualId = p.createVisualShape(shapeType=obj_type, halfExtents=size, rgbaColor=color)
        elif color is not None:
            visualId = p.createVisualShape(shapeType=obj_type, halfExtents=size, rgbaColor=color)

    elif obj_type == "random":
        obj = "%03d" % np.random.randint(1000)
        obj_id = p.loadURDF(f"random_urdfs/{obj}/{obj}.urdf", basePosition=position,
                            baseOrientation=p.getQuaternionFromEuler(rotation))
        return obj_id

    if with_link:
        obj_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1,
                                   basePosition=position, baseOrientation=p.getQuaternionFromEuler(rotation),
                                   linkMasses=[mass], linkCollisionShapeIndices=[collisionId],
                                   linkVisualShapeIndices=[visualId], linkPositions=[[0, 0, 0]],
                                   linkOrientations=[[0, 0, 0, 1]], linkInertialFramePositions=[[0, 0, 0]],
                                   linkInertialFrameOrientations=[[0, 0, 0, 1]], linkParentIndices=[0],
                                   linkJointTypes=[p.JOINT_FIXED], linkJointAxis=[[0, 0, 0]])
    else:
        obj_id = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collisionId, baseVisualShapeIndex=visualId,
                                   basePosition=position, baseOrientation=p.getQuaternionFromEuler(rotation))
        p.changeDynamics(obj_id, -1, rollingFriction=0.0005, spinningFriction=0.001)

    return obj_id


def create_arrow(p, from_loc, to_loc, color=[0.0, 1.0, 1.0, 0.75]):
    delta = (np.array(to_loc) - np.array(from_loc))
    length = np.linalg.norm(delta)
    r_x = -np.arctan2(np.linalg.norm([delta[0], delta[1]]), delta[2])
    r_y = 0
    r_z = -np.arctan2(delta[0], delta[1])

    baseVisualId = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01,
                                       rgbaColor=[0.0, 0.0, 1.0, 0.75])
    childVisualId = p.createVisualShape(shapeType=p.GEOM_CAPSULE, radius=0.0075, length=length,
                                        rgbaColor=color)
    tipVisualId = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01,
                                      rgbaColor=[1.0, 0.0, 1.0, 0.75])
    obj_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=baseVisualId,
                               basePosition=from_loc, baseOrientation=[0., 0., 0., 1], linkMasses=[-1, -1],
                               linkCollisionShapeIndices=[-1, -1], linkVisualShapeIndices=[childVisualId, tipVisualId],
                               linkPositions=[delta/2, delta],
                               linkOrientations=[p.getQuaternionFromEuler([r_x, r_y, r_z]), [0, 0, 0, 1]],
                               linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
                               linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
                               linkParentIndices=[0, 0], linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED],
                               linkJointAxis=[[0, 0, 0], [0, 0, 0]])
    return obj_id


def create_tabletop(p):
    objects = {}
    objects["base"] = create_object(p, p.GEOM_BOX, mass=0, size=[0.15, 0.15, 0.2],
                                    position=[0., 0., 0.2], color=[0.5, 0.5, 0.5, 1.0], with_link=True)
    objects["table"] = create_object(p, p.GEOM_BOX, mass=0, size=[0.7, 1, 0.2],
                                     position=[0.9, 0, 0.2], color=[0.9, 0.9, 0.9, 1.0])
    # walls
    objects["wall1"] = create_object(p, p.GEOM_BOX, mass=0, size=[0.7, 0.01, 0.05],
                                     position=[0.9, -1, 0.45], color=[1.0, 0.6, 0.6, 1.0])
    objects["wall2"] = create_object(p, p.GEOM_BOX, mass=0, size=[0.7, 0.01, 0.05],
                                     position=[0.9, 1, 0.45], color=[1.0, 0.6, 0.6, 1.0])
    objects["wall3"] = create_object(p, p.GEOM_BOX, mass=0, size=[0.01, 1, 0.05],
                                     position=[0.2, 0., 0.45], color=[1.0, 0.6, 0.6, 1.0])
    objects["wall4"] = create_object(p, p.GEOM_BOX, mass=0, size=[0.01, 1, 0.05],
                                     position=[1.6, 0., 0.45], color=[1.0, 0.6, 0.6, 1.0])
    return objects


def get_image(p, eye_position, target_position, up_vector, height, width):
    viewMatrix = p.computeViewMatrix(cameraEyePosition=eye_position,
                                     cameraTargetPosition=target_position,
                                     cameraUpVector=up_vector)
    projectionMatrix = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.75, farVal=5.5)
    _, _, rgb, depth, seg = p.getCameraImage(height=height, width=width, viewMatrix=viewMatrix,
                                             projectionMatrix=projectionMatrix)
    return rgb, depth, seg


def create_camera(p, position, rotation, static=True):
    baseCollision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
    targetCollision = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=0.005, height=0.01)
    baseVisual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[0, 0, 0, 1])
    targetVisual = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=0.005, length=0.01,
                                       rgbaColor=[0.8, 0.8, 0.8, 1.0])

    # base = create_object(obj_type=p.GEOM_SPHERE, size=0.1, position=position, rotation=rotation)
    # target = create_object(obj_T)
    mass = 0 if static else 0.1
    obj_id = p.createMultiBody(baseMass=mass,
                               baseCollisionShapeIndex=-1,
                               baseVisualShapeIndex=-1,
                               basePosition=position,
                               baseOrientation=p.getQuaternionFromEuler(rotation),
                               linkMasses=[mass, mass],
                               linkCollisionShapeIndices=[baseCollision, targetCollision],
                               linkVisualShapeIndices=[baseVisual, targetVisual],
                               linkPositions=[[0, 0, 0], [0.02, 0, 0]],
                               linkOrientations=[[0, 0, 0, 1], p.getQuaternionFromEuler([0., np.pi/2, 0])],
                               linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
                               linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
                               linkParentIndices=[0, 1],
                               linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED],
                               linkJointAxis=[[0, 0, 0], [0, 0, 0]])

    return obj_id


def get_image_from_cam(p, camera_id, height, width):
    cam_state = p.getLinkStates(camera_id, [0, 1])
    base_pos = cam_state[0][0]
    up_vector = Rotation.from_quat(cam_state[0][1]).as_matrix()[:, -1]
    target_pos = cam_state[1][0]
    target_vec = np.array(target_pos) - np.array(base_pos)
    target_vec = (target_vec / np.linalg.norm(target_vec))
    return get_image(base_pos+target_vec*0.04, base_pos+target_vec, up_vector, height, width)


def get_parameter_count(model):
    total_num = 0
    for param in model.parameters():
        total_num += param.shape.numel()
    return total_num


def print_module(module, name, space):
    L = len(name)
    line = " "*space+"-"*(L+4)
    print(line)
    print(" "*space+"  "+name+"  ")
    print(line)
    module_str = module.__repr__()
    print("\n".join([" "*space+mstr for mstr in module_str.split("\n")]))

#####################################


def decimal_to_binary(number_list, length=None):
    binaries = [format(x, "0"+str(length)+"b") for x in number_list]
    return binaries


def binary_to_decimal(x):
    '''
    Parameters
    ----------
    x : torch.Tensor
        N by D where N is the number of binary vectors

    Returns
    -------
    dec_tensor : torch.Tensor
        N length tensor that contains the decimal encodings
    '''
    N, D = x.shape
    dec_tensor = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    for i in reversed(range(D)):
        multiplier = 2**i
        dec_tensor += multiplier * x[:, D-i-1].int()
    return dec_tensor


def binary_tensor_to_str(x):
    return ["".join([str(x_ii.int().item()) for x_ii in x_i]) for x_i in x]


def str_to_binary_tensor(x):
    return torch.tensor([[int(i) for i in x_i] for x_i in x], dtype=torch.float)


def in_array(element, array):
    for i, e_i in enumerate(array):
        if element.is_equal(e_i):
            return True, i
    return False, None


def to_str_state(obj, rel, mask=None):
    if mask is None:
        mask = torch.ones(obj.shape[0])
    m = (mask == 1)
    n_obj = m.sum()
    mm = torch.outer(m, m)
    obj_str = "-".join(binary_tensor_to_str(obj.bernoulli()[m]))
    rel_str = "-".join([",".join(binary_tensor_to_str(r_i.bernoulli()[mm].reshape(n_obj, n_obj))) for r_i in rel])
    return obj_str + "_" + rel_str


def to_str_action(action):
    to_idx = int(torch.where(action[:, 0] > 0.5)[0])
    to_dx = int(action[to_idx, 1])
    to_dy = int(action[to_idx, 2])

    from_idx = torch.where(action[:, 0] < -0.5)[0]
    if len(from_idx) == 0:
        from_idx = to_idx
        from_dx = 0
        from_dy = 0
    else:
        from_idx = int(from_idx)
        from_dx = int(action[from_idx, 1])
        from_dy = int(action[from_idx, 2])
    act_str = f"{from_idx},{from_dx},{from_dy},{to_idx},{to_dx},{to_dy}"
    return act_str


def to_tensor_state(str_state):
    obj_str, rel_str = str_state.split("_")
    obj = str_to_binary_tensor(obj_str.split("-"))
    rel = torch.stack([str_to_binary_tensor(r_i.split(",")) for r_i in rel_str.split("-")])
    return obj, rel


def segment_img_with_mask(img, mask, valid_objects, window=64, padding=10, aug=False):
    if aug:
        T = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fillcolor=0)
    N_obj = mask.max()
    hw = window // 2
    segmented_imgs = []
    for i in range(N_obj+1):
        if i in valid_objects:
            mask_map = (mask == i)
            rows, cols = torch.where(mask_map)
            if len(rows) > 0:
                x_c = min(img.shape[1]-hw, max(hw, int(rows.float().mean())))
                y_c = min(img.shape[2]-hw, max(hw, int(cols.float().mean())))
                xmin, xmax = x_c-hw, x_c+hw
                ymin, ymax = y_c-hw, y_c+hw
                masked_img = (mask_map.float() * img)
                seg_img = masked_img[:, xmin:xmax, ymin:ymax] / 255.0
                img_h, img_w = seg_img.shape[1], seg_img.shape[2]
                x_ch = torch.arange(xmin, xmax).repeat_interleave(img_w, 0).reshape(1, img_h, img_w) / img.shape[1]
                y_ch = torch.arange(ymin, ymax).repeat(img_h, 1).unsqueeze(0) / img.shape[2]
                seg_img = torch.cat([seg_img, x_ch, y_ch], dim=0)
                if aug:
                    seg_img = T(seg_img)
                segmented_imgs.append(seg_img)
    segmented_imgs = torch.stack(segmented_imgs)
    return segmented_imgs


def segment_img_with_mask_old(img, mask, valid_objects):
    N_obj = mask.max()
    segmented_imgs = []
    for i in range(N_obj+1):
        if i in valid_objects:
            obj_mask = (mask == i)
            if obj_mask.any():
                obj_mask = obj_mask.float()
                seg_img = (img * obj_mask) / 255.0
                segmented_imgs.append(seg_img)
    segmented_imgs = torch.stack(segmented_imgs)
    return segmented_imgs


def preprocess(state, segmentation, valid_objects, max_pad, aug, old):
    if old:
        seg_a = segment_img_with_mask_old(state, segmentation, valid_objects)
    else:
        seg_a = segment_img_with_mask(state, segmentation, valid_objects, aug=aug)
    n_seg, ch, h, w = seg_a.shape
    n_seg = min(n_seg, max_pad)
    padded = torch.zeros(max_pad, ch, h, w)
    padded[:n_seg] = seg_a[:n_seg]
    pad_mask = torch.zeros(max_pad)
    pad_mask[:n_seg] = 1.0
    return padded, pad_mask


def normalize_depth_img(img):
    vmin = 0.5
    vmax = 1
    if img.min() < 0.5:
        print("chaos is here.", img.min())
    img_n = (((img - vmin) / (vmax - vmin))*255).astype(np.uint8)
    return img_n


def state_to_tensor(state_tuple, valid_objects, num_objects, is_old, is_aug=False):
    _, depth_img, seg = state_tuple
    depth_img = normalize_depth_img(depth_img)
    depth_img = torch.tensor(depth_img).unsqueeze(0)
    seg = torch.tensor(seg)
    padded, pad_mask = preprocess(depth_img, seg, valid_objects, num_objects, old=is_old, aug=is_aug)
    return padded, pad_mask


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def sample_prediction(prob, mask, sample_size=100):
    valid_tokens = torch.stack([p_i for p_i, m_i in zip(prob, mask) if m_i > 0.5])
    possible_symbols = []
    for _ in range(sample_size):
        sample = valid_tokens.bernoulli()
        possible_symbols.append(sample)
    possible_symbols = torch.stack(possible_symbols).unique(dim=0)
    sampled_symbols = []
    for p_i in possible_symbols:
        current_sample = []
        it = 0
        for i, m_i in enumerate(mask):
            if m_i > 0.5:
                current_sample.append(p_i[it])
                it += 1
            else:
                current_sample.append(prob[i].round())
        current_sample = torch.stack(current_sample)
        sampled_symbols.append(current_sample)
    sampled_symbols = torch.stack(sampled_symbols)
    return sampled_symbols


def wandb_finalize():
    wandb.finish()


def upload_dataset_to_wandb(name, path):
    with zipfile.ZipFile(f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(path):
            if file != ".DS_Store":
                zipf.write(os.path.join(path, file), arcname=file)
    wandb.init(project="multideepsym", entity="colorslab")
    artifact = wandb.Artifact(name, type="dataset")
    artifact.add_file(f"{name}.zip")
    wandb.log_artifact(artifact)
    wandb_finalize()
    os.remove(f"{name}.zip")


def get_dataset_from_wandb(name):
    artifact = wandb.use_artifact(f"colorslab/multideepsym/{name}:latest", type="dataset")
    artifact_dir = artifact.download()
    archive = zipfile.ZipFile(os.path.join(artifact_dir, f"{name}.zip"), "r")
    archive.extractall(os.path.join("data", name))
    archive.close()
    os.remove(os.path.join(artifact_dir, f"{name}.zip"))


def change_rows(matrix, lst):
    return matrix[lst, :]


def change_columns(matrix, lst):
    return matrix[:, lst]


def permute(nums):
    return list(permutations(nums))


def change_rows_and_cols(matrix, ary):
    new_matrix = change_rows(matrix, ary)
    new_matrix = change_columns(new_matrix, ary)
    return new_matrix


def permute_symbols(obj_pre, rel_pre, action, obj_post, rel_post, perm):
    obj_pre = tuple(binary_tensor_to_str(change_rows(obj_pre, perm)))
    rel_pre_permuted = []
    rel_post_permuted = []
    for k in range(len(rel_pre)):
        rel_pre_permuted.append(tuple(binary_tensor_to_str(change_rows_and_cols(rel_pre[k], perm))))
        rel_post_permuted.append(tuple(binary_tensor_to_str(change_rows_and_cols(rel_post[k], perm))))
    rel_pre_permuted = tuple(rel_pre_permuted)
    rel_post_permuted = tuple(rel_post_permuted)
    action = tuple(binary_tensor_to_str(change_rows(action, perm)))
    obj_post = tuple(binary_tensor_to_str(change_rows(obj_post, perm)))
    return obj_pre, rel_pre_permuted, action, obj_post, rel_post_permuted
