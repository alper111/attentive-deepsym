from copy import deepcopy

import rospy
from baxter_core_msgs.msg import JointCommand, EndEffectorCommand
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from std_msgs.msg import Bool, Header
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from sensor_msgs.msg import JointState


class BaxterRobot:

    def __init__(self, arm, rate=100):
        self.rate = rospy.Rate(rate)

        self.name = arm
        self._joint_angle = {}
        self._joint_velocity = {}
        self._joint_effort = {}
        self._cartesian_pose = {}
        self._cartesian_velocity = {}
        self._cartesian_effort = {}
        self._joint_names = ["_s0", "_s1", "_e0", "_e1", "_w0", "_w1", "_w2"]
        self._joint_names = [arm+x for x in self._joint_names]
        ns = "/robot/limb/" + arm + "/"
        iksvc_ns = "/ExternalTools/" + arm + "/PositionKinematicsNode/IKService"
        self.iksvc = rospy.ServiceProxy(iksvc_ns, SolvePositionIK)
        rospy.wait_for_service(iksvc_ns)
        print("IK service loaded.")
        self._command_msg = JointCommand()
        self._robot_state = rospy.Publisher("/robot/set_super_enable", Bool, queue_size=10)
        self._pub_joint_cmd = rospy.Publisher(ns+"joint_command", JointCommand, queue_size=10)
        self._joint_state_sub = rospy.Subscriber("/robot/joint_states", JointState, self._fill_joint_state)
        self._gripper_cmd = rospy.Publisher("/robot/end_effector/left_gripper/command", EndEffectorCommand, queue_size=10)
        rospy.sleep(1.0)
        calibrate_msg = EndEffectorCommand()
        calibrate_msg.id = 65538
        calibrate_msg.command = "calibrate"
        self._gripper_cmd.publish(calibrate_msg)

    def open_gripper(self):
        msg = EndEffectorCommand()
        msg.id = 65538
        msg.command = "go"
        msg.args = "{\"position\": 100.0}"
        self._gripper_cmd.publish(msg)

    def close_gripper(self):
        msg = EndEffectorCommand()
        msg.id = 65538
        msg.command = "go"
        msg.args = "{\"position\": 0.0}"
        self._gripper_cmd.publish(msg)

    def joint_angle(self):
        return deepcopy(self._joint_angle)

    def set_robot_state(self, state):
        msg = Bool()
        msg.data = state
        self._robot_state.publish(msg)

    def set_cartesian_position(self, position, orientation):
        hdr = Header(stamp=rospy.Time.now(), frame_id="base")
        msg = PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(
                    x=position[0],
                    y=position[1],
                    z=position[2]
                ),
                orientation=Quaternion(
                    x=orientation[0],
                    y=orientation[1],
                    z=orientation[2],
                    w=orientation[3]
                )
            )
        )
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(msg)
        resp = self.iksvc(ikreq)
        if resp.isValid[0]:
            self.move_to_joint_position(
                {
                    self.name+"_s0": resp.joints[0].position[0],
                    self.name+"_s1": resp.joints[0].position[1],
                    self.name+"_e0": resp.joints[0].position[2],
                    self.name+"_e1": resp.joints[0].position[3],
                    self.name+"_w0": resp.joints[0].position[4],
                    self.name+"_w1": resp.joints[0].position[5],
                    self.name+"_w2": resp.joints[0].position[6],
                }
            )
        else:
            print(resp)

    def set_joint_position(self, positions):
        self._command_msg.names = list(positions.keys())
        self._command_msg.command = list(positions.values())
        self._command_msg.mode = JointCommand.POSITION_MODE
        self._pub_joint_cmd.publish(self._command_msg)

    def set_joint_velocity(self, velocities):
        self._command_msg.names = list(velocities.keys())
        self._command_msg.command = list(velocities.values())
        self._command_msg.mode = JointCommand.VELOCITY_MODE
        self._pub_joint_cmd.publish(self._command_msg)

    def move_to_joint_position(self, positions, timeout=15.0):
        current_angle = self.joint_angle()
        end_time = rospy.get_time() + timeout

        # update the target based on the current location
        # if you use this instead of positions, the jerk
        # will be smaller.
        def current_target():
            for joint in positions:
                current_angle[joint] = 0.012488 * positions[joint] + 0.98751 * current_angle[joint]
            return current_angle

        def difference():
            diffs = []
            for joint in positions:
                diffs.append(abs(positions[joint] - self._joint_angle[joint]))
            return diffs

        while any(diff > 0.008726646 for diff in difference()) and rospy.get_time() < end_time:
            self.set_joint_position(current_target())
            self.rate.sleep()
        return all(diff < 0.008726646 for diff in difference())

    def move_to_neutral(self):
        angles = dict(list(zip(self._joint_names, [0.0, -0.55, 0.0, 0.75, 0.0, 1.26, 0.0])))
        return self.move_to_joint_position(angles)

    def move_to_zero(self):
        angles = dict(list(zip(self._joint_names, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))
        return self.move_to_joint_position(angles)

    def _fill_joint_state(self, msg):
        for idx, name in enumerate(msg.name):
            if name in self._joint_names:
                self._joint_angle[name] = msg.position[idx]
                self._joint_velocity[name] = msg.velocity[idx]
                self._joint_effort[name] = msg.effort[idx]
