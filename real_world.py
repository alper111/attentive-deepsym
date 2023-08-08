import rospy
import baxter
import numpy as np
np.float = np.float64
import torch
import ros_numpy
from sklearn.cluster import KMeans, DBSCAN
from sensor_msgs.msg import PointCloud2
import matplotlib.pyplot as plt

from models import load_ckpt
import mcts


def camera_to_robot(point):
    return [0.92 - point[1], point[0]+0.054, 0.9 - point[2]]


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


if __name__ == "__main__":
    model, _ = load_ckpt("atten_o234_5")
    model.freeze()
    forward_fn = mcts.SubsymbolicForwardModel(model, obj_relative=True)

    point_cloud_topic = "/camera/depth/color/points"
    rgb_topic = "/camera/color/image_raw"
    depth_topic = "/camera/depth/image_rect_raw"
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

    for i in range(10):
        msg = rospy.wait_for_message(point_cloud_topic, PointCloud2)
        data = ros_numpy.numpify(msg)
        data = ros_numpy.point_cloud2.split_rgb_field(data)
        xyz = np.array([[-x, y, z] for (x, y, z, _, _, _) in data])
        rgb = np.array([[r, g, b] for (_, _, _, r, g, b) in data])
        rgb = rgb / 255.0
        x_mask = np.logical_and(xyz[:, 0] > -0.2, xyz[:, 0] < 0.5)
        y_mask = np.logical_and(xyz[:, 1] > 0, xyz[:, 1] < 0.35)
        mask = np.logical_and(x_mask, y_mask)
        points = np.concatenate((xyz[mask], rgb[mask]), axis=1)

        clustering = DBSCAN(eps=0.1, min_samples=200).fit(points[:, [0, 1, 3, 4, 5]])
        # print(clustering.cluster_centers_)
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

        plt.scatter(points[:, 0], points[:, 1], c=rgb[mask], cmap="tab10")
        if red_obj is not None:
            plt.scatter(red_obj[0], red_obj[1], c="red", s=100, marker="x")
        if green_obj is not None:
            plt.scatter(green_obj[0], green_obj[1], c="green", s=100, marker="x")
        if blue_obj is not None:
            plt.scatter(blue_obj[0], blue_obj[1], c="blue", s=100, marker="x")
        if yellow_obj is not None:
            plt.scatter(yellow_obj[0], yellow_obj[1], c="yellow", s=100, marker="x")
        plt.show()
        print(yellow_obj, blue_obj, green_obj, red_obj)
        # robot_stack(camera_to_robot(red_obj), camera_to_robot(blue_obj), p2y=0.075)
        # robot_stack(camera_to_robot(green_obj), camera_to_robot(blue_obj), p2y=-0.075)
        # robot_stack(camera_to_robot(yellow_obj), camera_to_robot(blue_obj), p1d=-0.03, p2d=0.1)

        o1 = torch.tensor(camera_to_robot(yellow_obj))
        o2 = torch.tensor(camera_to_robot(blue_obj))
        o3 = torch.tensor(camera_to_robot(green_obj))
        o4 = torch.tensor(camera_to_robot(red_obj))
        o1 = torch.cat([o1, torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 0, 0])]).float()
        o2 = torch.cat([o2, torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 0, 0])]).float()
        o3 = torch.cat([o3, torch.tensor([1, 0, 1, 0, 1, 0, 0, 0, 0, 1])]).float()
        o4 = torch.cat([o4, torch.tensor([1, 0, 1, 0, 1, 0, 0, 0, 0, 1])]).float()
        state = torch.stack([o1, o2, o3, o4], dim=0)
        goal = state.clone()
        goal[0, :3] = o2[:3] + torch.tensor([0.0, 0.0, 0.075])
        goal[1, :3] = o2[:3]
        goal[2, :3] = o2[:3]
        goal[3, :3] = o2[:3] + torch.tensor([0.0, 0.075, 0.0])
        print(state.shape)
        print(goal.shape)
        obj_dict = {0: "yellow", 1: "blue", 2: "green", 3: "red"}
        init = mcts.SubsymbolicState(state, goal)
        node = mcts.MCTSNode(node_id=0, parent=None, state=init, forward_fn=forward_fn)
        children = node.run(5000, 600, default_depth_limit=1)
        _, plan, _, _ = node.plan()
        print(plan)
        execute = input("Execute plan? (y/n): ")
        if execute == "y":
            actions = plan.split("_")
            for action in actions:
                action = action.split(",")
                obj1 = int(action[0])
                obj2 = int(action[3])
                dy1 = float(action[2]) * 0.075
                dy2 = float(action[5]) * 0.075
                robot_stack(camera_to_robot(o1), camera_to_robot(o2), p1y=dy1, p2y=dy2)
        else:
            print("Plan not executed.")

    # has_red = rgb[:, 0] > 150
    # no_red = rgb[:, 0] < 50
    # has_green = rgb[:, 1] > 100
    # no_green = rgb[:, 1] < 50
    # has_blue = rgb[:, 2] > 85
    # no_blue = rgb[:, 2] < 50
    # is_blue = np.logical_and(has_blue, no_red)
    # is_red = np.logical_and(has_red, no_green)
    # is_yellow = np.logical_and(np.logical_and(has_red, has_green), no_blue)
    # is_white = np.logical_and(np.logical_and(has_red, has_green), has_blue)
    # plt.scatter(xyz[mask, 0], xyz[mask, 1], c=rgb[mask]/255.0)
    # mask = np.logical_and(mask, is_white)
    # plt.scatter(xyz[mask, 0], xyz[mask, 1], c=[0., 1., 1])
    # plt.show()
