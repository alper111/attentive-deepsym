import time
import threading

import rospy
import baxter
import numpy as np
np.float = np.float64
import torch
import ros_numpy
from sklearn.cluster import KMeans, DBSCAN
from sensor_msgs.msg import PointCloud2, Image
import matplotlib.pyplot as plt
import cv2
import dearpygui.dearpygui as dpg

from models import load_ckpt
from environment import EmptyEnv
import mcts
import utils


POINT_CLOUD_TOPIC = "/camera/depth/color/points"
RGB_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/depth/image_rect_raw"


def camera_to_robot(point):
    return [0.87 - point[1], point[0]+0.054, 0.9 - point[2] + 0.48]


def robot_stack(p1, p2, p1y=0.0, p1d=0.0, p2y=0.0, p2d=0.0, sleep_dur=0.5):
    p1r = p1.copy()
    p2r = p2.copy()
    p1r[2] = 0.15 + p1d
    p2r[2] = 0.15 + p2d
    p1r[1] += p1y
    p2r[1] += p2y

    robot.open_gripper()
    rospy.sleep(sleep_dur)
    robot.set_cartesian_position(p1r, [-0.7, 0.7, 0.0, 0.0])
    rospy.sleep(sleep_dur)
    p1r[2] = -0.12 + p1d
    robot.set_cartesian_position(p1r, [-0.7, 0.7, 0.0, 0.0])
    rospy.sleep(sleep_dur)
    robot.close_gripper()
    rospy.sleep(sleep_dur)
    p1r[2] = 0.15
    robot.set_cartesian_position(p1r, [-0.7, 0.7, 0.0, 0.0])
    rospy.sleep(sleep_dur)
    robot.set_cartesian_position(p2r, [-0.7, 0.7, 0.0, 0.0])
    rospy.sleep(sleep_dur)
    p2r[2] = -0.05 + p2d
    robot.set_cartesian_position(p2r, [-0.7, 0.7, 0.0, 0.0])
    rospy.sleep(sleep_dur)
    robot.open_gripper()
    rospy.sleep(sleep_dur)
    p2r[2] = 0.15 + p2d
    robot.set_cartesian_position(p2r, [-0.7, 0.7, 0.0, 0.0])
    rospy.sleep(sleep_dur)
    robot.set_cartesian_position([0.50, 0.52, 0.06], [-0.7, 0.7, 0.0, 0.0])
    rospy.sleep(sleep_dur)


def initialize_node():
    rospy.init_node("demo", anonymous=True)
    rospy.sleep(1.0)
    robot = baxter.BaxterRobot(rate=100, arm="left")
    robot.set_robot_state(True)
    robot.close_gripper()
    rospy.sleep(1.0)
    robot.open_gripper()
    rospy.sleep(1.0)
    robot.set_cartesian_position([0.50, 0.52, 0.06], [0.7, -0.7, 0.0, 0.0])
    rospy.sleep(1.0)
    return robot


def detect_and_predict(env, forward_fn):
    start = time.time()
    msg = rospy.wait_for_message(POINT_CLOUD_TOPIC, PointCloud2)
    data = ros_numpy.numpify(msg)
    data = ros_numpy.point_cloud2.split_rgb_field(data)
    end = time.time()
    print(f"Parse time: {end - start}")
    start = time.time()
    xyz = np.array([[-x, y, z] for (x, y, z, _, _, _) in data])
    rgb = np.array([[r, g, b] for (_, _, _, r, g, b) in data])
    rgb = rgb / 255.0
    x_mask = np.logical_and(xyz[:, 0] > -0.2, xyz[:, 0] < 0.5)
    y_mask = np.logical_and(xyz[:, 1] > 0, xyz[:, 1] < 0.35)
    mask = np.logical_and(x_mask, y_mask)
    points = np.concatenate((xyz[mask], rgb[mask]), axis=1)
    clustering = DBSCAN(eps=0.1, min_samples=200).fit(points[:, [0, 1, 3, 4, 5]])
    end = time.time()
    print(f"Clustering time: {end - start}")

    start = time.time()
    labels = clustering.labels_
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    red_obj, green_obj, blue_obj, yellow_obj = None, None, None, None
    for label in range(num_clusters):
        center = points[labels == label].mean(axis=0)
        print(label, center)
        if center[3] > 0.35 and center[4] < 0.3 and center[5] < 0.3:
            red_obj = center
        elif center[3] < 0.7 and center[4] > 0.5 and center[5] > 0.4 and center[5] < 0.7:
            green_obj = center
        elif center[3] > 0.8 and center[4] > 0.7 and center[5] < 0.5:
            yellow_obj = center
        elif center[3] < 0.1 and center[4] > 0.6 and center[5] > 0.7:
            blue_obj = center
    o1 = torch.tensor(camera_to_robot(red_obj))
    o2 = torch.tensor(camera_to_robot(green_obj))
    o3 = torch.tensor(camera_to_robot(blue_obj))
    o4 = torch.tensor(camera_to_robot(yellow_obj))
    o1[2] = 0.45
    o2[2] = 0.45
    o3[2] = 0.425
    o4[2] = 0.425
    print(f"Rest time: {time.time() - start}")
    print(f"Red: {o1}")
    print(f"Green: {o2}")
    print(f"Blue: {o3}")
    print(f"Yellow: {o4}")
    start = time.time()
    env.reset_objects()
    env.add_object(position=o1, type=1, color=[1, 0, 0, 1])
    env.add_object(position=o2, type=1, color=[0, 1, 0, 1])
    env.add_object(position=o3, type=4, color=[0, 0, 1, 1])
    env.add_object(position=o4, type=4, color=[1, 1, 0, 1])
    env._step(240)
    end = time.time()
    print(f"Env time: {end - start}")

    o1 = torch.cat([o1, torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 0, 0])]).float()
    o2 = torch.cat([o2, torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 0, 0])]).float()
    o3 = torch.cat([o3, torch.tensor([1, 0, 1, 0, 1, 0, 0, 0, 0, 1])]).float()
    o4 = torch.cat([o4, torch.tensor([1, 0, 1, 0, 1, 0, 0, 0, 0, 1])]).float()
    state = torch.stack([o1, o2, o3, o4], dim=0)
    init = mcts.SubsymbolicState(state, state)
    state1 = forward_fn(init, action="0,0,0,2,0,1", obj_relative=True)
    state2 = forward_fn(state1, action="1,0,0,2,0,-1", obj_relative=True)
    state3 = forward_fn(state2, action="3,0,0,2,0,0", obj_relative=True)
    for o_i, o_f in zip(state3.state, state):
        env.add_arrow(o_i[:3], o_f[:3], color=[0, 1, 1, 0.75])
    update_sim_frame(env)
    robot_stack(camera_to_robot(red_obj), camera_to_robot(blue_obj), p2y=0.075)
    robot_stack(camera_to_robot(green_obj), camera_to_robot(blue_obj), p2y=-0.075)
    robot_stack(camera_to_robot(yellow_obj), camera_to_robot(blue_obj), p1d=-0.03, p2d=0.1)
    return red_obj, green_obj, blue_obj, yellow_obj


def update_front_frame(cam, width=640, height=480):
    start = time.time()
    while True:
        if time.time() - start > 1/60:
            start = time.time()
            ret, fr = cam.read()
            if ret:
                fr = cv2.resize(fr, (width, height))
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                dpg.set_value("front", fr)


def update_rgb_frame(msg):
    rgb_frame = ros_numpy.numpify(msg)
    rgb_frame = cv2.resize(rgb_frame, (320, 240))
    rgb_frame = rgb_frame.astype(np.float32) / 255.0
    dpg.set_value("rgb", rgb_frame)


def update_depth_frame(msg):
    depth_frame = ros_numpy.numpify(msg)
    depth_frame = cv2.resize(depth_frame, (320, 240))
    depth_frame = depth_frame.astype(np.float32)
    depth_frame = np.clip(depth_frame, 900, 1100)
    depth_frame = (depth_frame - depth_frame.min()) / (depth_frame.max() - depth_frame.min())
    depth_frame = np.expand_dims(depth_frame, axis=-1)
    depth_frame = np.concatenate([depth_frame, depth_frame, depth_frame], axis=-1)
    dpg.set_value("depth", depth_frame)


def update_sim_frame(env, width=640, height=480):
    rgb, _, _ = utils.get_image(env._p, [2.1, 0.0, 1.5], [0.8, 0, 0.5],
                                up_vector=[0, 0, 1], width=width, height=height)
    rgb = rgb.astype(np.float32) / 255.0
    dpg.set_value("sim", rgb)


if __name__ == "__main__":
    model, _ = load_ckpt("atten_o234_5")
    model.freeze()
    forward_fn = mcts.SubsymbolicForwardModel(model, obj_relative=True)
    # create the context and the viewport
    dpg.create_context()
    dpg.create_viewport(title="Attentive DeepSym Visualization")
    robot = initialize_node()
    env = EmptyEnv(gui=1)

    front_frame = np.zeros((480, 640, 3), dtype=np.float32)
    rgb_frame = np.zeros((240, 320, 3), dtype=np.float32)
    depth_frame = np.zeros((240, 320, 1), dtype=np.float32)
    sim_frame = np.zeros((480, 640, 4), dtype=np.float32)

    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(width=640, height=480, default_value=front_frame,
                            format=dpg.mvFormat_Float_rgb, tag="front")
        dpg.add_raw_texture(width=320, height=240, default_value=rgb_frame,
                            format=dpg.mvFormat_Float_rgb, tag="rgb")
        dpg.add_raw_texture(width=320, height=240, default_value=depth_frame,
                            format=dpg.mvFormat_Float_rgb, tag="depth")
        dpg.add_raw_texture(width=640, height=480, default_value=sim_frame,
                            format=dpg.mvFormat_Float_rgba, tag="sim")

    with dpg.window(tag="Main Window", label="Main Window"):
        with dpg.table(header_row=False):
            dpg.add_table_column(label="Left Col", tag="LeftCol", width_fixed=True)
            dpg.add_table_column(label="Right Col", tag="RightCol", width_fixed=True)

            with dpg.table_row():
                with dpg.table_cell():
                    dpg.add_image("front", label="Front Camera")
                with dpg.table_cell():
                    dpg.add_image("sim", label="Simulation")
                    dpg.add_button(label="Detect and Predict", callback=lambda: detect_and_predict(env, forward_fn))

            with dpg.table_row():
                with dpg.table(header_row=False):
                    dpg.add_table_column(label="Header 1", tag="RgbCol", width_fixed=True)
                    dpg.add_table_column(label="Header 2", tag="DepthCol", width_fixed=True)

                    with dpg.table_row():
                        with dpg.table_cell():
                            dpg.add_text("RGB Image")
                            dpg.add_image("rgb", label="RGB Image")

                        with dpg.table_cell():
                            dpg.add_text("Depth Image")
                            dpg.add_image("depth", label="Depth Image")

    # initialize cameras and frames
    front_cam = cv2.VideoCapture(8)
    threading.Thread(target=update_front_frame, args=(front_cam,)).start()
    rgb_sub = rospy.Subscriber(RGB_TOPIC, Image, callback=update_rgb_frame)
    depth_sub = rospy.Subscriber(DEPTH_TOPIC, Image, callback=update_depth_frame)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("Main Window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()
