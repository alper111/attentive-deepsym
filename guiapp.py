import time
import multiprocessing

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from environment import EmptyEnv
from utils import get_image


def update_camera_frame(cam, frame, width=640, height=480):
    ret, fr = cam.read()
    fr = cv2.resize(fr, (width, height))
    if ret:
        frame[:, :, :] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def update_sim_image(env, frame):
    rgb, _, _ = get_image(env._p, [1.5, 0.0, 1.5], [0.9, 0, 0.4],
                          up_vector=[0, 0, 1], width=320, height=240)
    frame[:, :, :] = rgb[:, :, :3] / 255.0


dpg.create_context()
dpg.create_viewport(title="Attentive DeepSym Visualization")
depth_cam = cv2.VideoCapture(6)
front_cam = cv2.VideoCapture(8)
_, front_frame = front_cam.read()
front_frame = cv2.cvtColor(front_frame, cv2.COLOR_BGR2RGB)
front_frame = front_frame.astype(np.float32) / 255.0
_, depth_frame = depth_cam.read()
depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2RGB)
depth_frame = depth_frame.astype(np.float32) / 255.0
depth_frame = np.zeros((240, 320, 3), dtype=np.float32)
sim_frame = np.zeros((240, 320, 3), dtype=np.float32)

with dpg.texture_registry(show=False):
    dpg.add_raw_texture(width=640, height=480, default_value=front_frame,
                        format=dpg.mvFormat_Float_rgb, tag="front")
    dpg.add_raw_texture(width=320, height=240, default_value=depth_frame,
                        format=dpg.mvFormat_Float_rgb, tag="depth")
    dpg.add_raw_texture(width=320, height=240, default_value=sim_frame,
                        format=dpg.mvFormat_Float_rgb, tag="sim")

with dpg.window(tag="Main Window", label="Main Window"):
    with dpg.table(header_row=False):
        dpg.add_table_column(label="Header 1", tag="FrontCol", width_fixed=True)
        dpg.add_table_column(label="Header 2", tag="SimCol", width_fixed=True)

        with dpg.table_row():
            dpg.add_image("front")
            with dpg.table_cell():
                dpg.add_image("depth")
                dpg.add_image("sim")

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Main Window", True)
env = EmptyEnv(gui=1)
env.add_object(position=[0.7, 0.0, 0.5], type=1, color=[1, 1, 0, 1])
env.add_object(position=[0.7, 0.0, 0.5], type=4, color=[0, 1, 1, 1])
env._step(240)
start = time.time()
while dpg.is_dearpygui_running():
    if time.time() - start > 1/60:
        start = time.time()
        # env._step()
        update_camera_frame(front_cam, front_frame)
        update_camera_frame(depth_cam, depth_frame, width=320, height=240)
        update_sim_image(env, sim_frame)
    dpg.render_dearpygui_frame()
depth_cam.release()
front_cam.release()
dpg.destroy_context()
