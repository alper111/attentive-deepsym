import time

import numpy as np


class Manipulator:
    def __init__(self, p, path, position=(0, 0, 0), orientation=(0, 0, 0, 1), ik_idx=-1):
        self._p = p
        self._timestep = self._p.getPhysicsEngineParameters()["fixedTimeStep"]
        self._freq = int(1. / self._timestep)
        self._total_steps = 0
        self._last_time = 0.0
        self.id = self._p.loadURDF(
            fileName=path,
            basePosition=position,
            baseOrientation=orientation,
            flags=p.URDF_USE_SELF_COLLISION)
        self.ik_idx = ik_idx
        self.joints = []
        self.names = []
        self.forces = []
        self.fixed_joints = []
        self.fixed_names = []
        for i in range(self._p.getNumJoints(self.id)):
            info = self._p.getJointInfo(self.id, i)
            if info[2] != self._p.JOINT_FIXED:
                self.joints.append(i)
                self.names.append(info[1])
                self.forces.append(info[10])
            else:
                self.fixed_joints.append(i)
                self.fixed_names.append(info[1])
        self.joints = tuple(self.joints)
        self.names = tuple(self.names)
        self.num_joints = len(self.joints)
        self.debug_params = []
        self.child = None
        self.constraints = []
        for j in self.joints:
            self._p.enableJointForceTorqueSensor(self.id, j, 1)
        self.force_stop_threshold = -7.5

    def attach(self, child_body, parent_link, position=(0, 0, 0), orientation=(0, 0, 0, 1), max_force=None):
        self.child = child_body
        constraint_id = self._p.createConstraint(
            parentBodyUniqueId=self.id,
            parentLinkIndex=parent_link,
            childBodyUniqueId=self.child.id,
            childLinkIndex=-1,
            jointType=self._p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=position,
            childFrameOrientation=orientation)
        if max_force is not None:
            self._p.changeConstraint(constraint_id, maxForce=max_force)
        self.constraints.append(constraint_id)

    def set_cartesian_position(self, position, orientation=None, t=None, sleep=False, traj=False):
        target_joints = self._p.calculateInverseKinematics(
            bodyUniqueId=self.id,
            endEffectorLinkIndex=self.ik_idx,
            targetPosition=position,
            targetOrientation=orientation)
        self.set_joint_position(target_joints[:-2], t=t, sleep=sleep, traj=traj)

    def move_in_cartesian(self, position, orientation, t=1.0, sleep=False, ignore_force=False):
        N = int(t * 240)

        current_position, current_orientation = self.get_tip_pose()

        position_traj = np.linspace(current_position, position, N+1)[1:]
        orientation_traj = np.linspace(current_orientation, orientation, N+1)[1:]
        running_force_feedback = np.zeros(8, dtype=np.float32)
        for i, p_i in enumerate(position_traj):
            target_joints = self._p.calculateInverseKinematics(
                bodyUniqueId=self.id,
                endEffectorLinkIndex=self.ik_idx,
                targetPosition=p_i,
                targetOrientation=orientation_traj[i])
            self.set_joint_position(target_joints[:-2], t=1/240, sleep=sleep)
            force_feedback = np.array(self.get_joint_forces())
            running_force_feedback = 0.9 * running_force_feedback + 0.1 * force_feedback
            if not ignore_force:
                if running_force_feedback[5] < self.force_stop_threshold:
                    # print("="*100)
                    for j in range(N//20):
                        target_joints = self._p.calculateInverseKinematics(
                            bodyUniqueId=self.id,
                            endEffectorLinkIndex=self.ik_idx,
                            targetPosition=position_traj[i-j],
                            targetOrientation=orientation)
                        self.set_joint_position(target_joints[:-2], t=1/240, sleep=sleep)
                        force_feedback = np.array(self.get_joint_forces())
                    break

    def set_joint_position(self, position, velocity=None, t=None, sleep=False, traj=False):
        assert len(self.joints) > 0
        if traj:
            assert (t is not None)
            N = int(t * 240)
            current_position = self.get_joint_position()[:-2]
            trajectory = np.linspace(current_position, position, N)
            running_force_feedback = np.zeros(8, dtype=np.float32)
            for i, t_i in enumerate(trajectory):
                self._p.setJointMotorControlArray(
                    bodyUniqueId=self.id,
                    jointIndices=self.joints[:-2],
                    controlMode=self._p.POSITION_CONTROL,
                    targetPositions=t_i,
                    forces=self.forces[:-2])
                self._p.stepSimulation()
                # print("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % tuple(self.get_joint_forces()))
                force_feedback = np.array(self.get_joint_forces())
                running_force_feedback = 0.9 * running_force_feedback + 0.1 * force_feedback
                if running_force_feedback[5] < self.force_stop_threshold:
                    # print("="*100)
                    for j in range(N//20):
                        self._p.setJointMotorControlArray(
                            bodyUniqueId=self.id,
                            jointIndices=self.joints[:-2],
                            controlMode=self._p.POSITION_CONTROL,
                            targetPositions=trajectory[i-j],
                            forces=self.forces[:-2])
                        self._p.stepSimulation()
                        if sleep:
                            time.sleep(self._timestep)
                        force_feedback = np.array(self.get_joint_forces())
                    break
                if sleep:
                    time.sleep(self._timestep)

        else:
            if velocity is not None:
                self._p.setJointMotorControlArray(
                    bodyUniqueId=self.id,
                    jointIndices=self.joints[:-2],
                    controlMode=self._p.POSITION_CONTROL,
                    targetPositions=position,
                    targetVelocities=velocity,
                    forces=self.forces[:-2])
            else:
                self._p.setJointMotorControlArray(
                    bodyUniqueId=self.id,
                    jointIndices=self.joints[:-2],
                    controlMode=self._p.POSITION_CONTROL,
                    targetPositions=position,
                    forces=self.forces[:-2])
            self._waitsleep(t, sleep)

    def set_joint_velocity(self, velocity, t=None, sleep=False):
        assert len(self.joints) > 0
        self._p.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.joints,
            controlMode=self._p.VELOCITY_CONTROL,
            targetVelocities=velocity,
            forces=self.forces)
        self._waitsleep(t, sleep)

    def set_joint_torque(self, torque, t=None, sleep=False):
        assert len(self.joints) > 0
        self._p.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.joints,
            controlMode=self._p.TORQUE_CONTROL,
            forces=torque)
        self._waitsleep(t, sleep)

    # this is only specific to UR10 + panda_gripper
    def open_gripper(self, t=None, sleep=False):
        self._set_gripper([0.04, 0.04], t=t, sleep=sleep)

    # this is only specific to UR10 + panda_gripper
    def close_gripper(self, t=None, sleep=False):
        self._set_gripper([0.0, 0.0], t=t, sleep=sleep)

    # this is only specific to UR10 + panda_gripper
    def _set_gripper(self, target_position, t=None, sleep=False):
        self._p.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.joints[-2:],
            controlMode=self._p.POSITION_CONTROL,
            targetPositions=target_position,
            forces=[15, 15])
        self._waitsleep(t, sleep)

    # TODO: make this only joint position, joint velocity etc.
    def get_joint_states(self):
        return self._p.getJointStates(self.id, self.joints)

    def get_joint_position(self):
        joint_states = self.get_joint_states()
        return [joint[0] for joint in joint_states]

    def get_joint_forces(self):
        joint_states = self.get_joint_states()
        return [joint[2][2] for joint in joint_states]

    # of IK link
    def get_tip_pose(self):
        result = self._p.getLinkState(self.id, self.ik_idx)
        return result[0], result[1]

    def add_debug_param(self):
        current_angle = [j[0] for j in self.get_joint_states()]
        for i in range(self.num_joints):
            joint_info = self._p.getJointInfo(self.id, self.joints[i])
            low, high = joint_info[8:10]
            self.debug_params.append(self._p.addUserDebugParameter(self.names[i].decode("utf-8"),
                                     low, high, current_angle[i]))

    def update_debug(self):
        target_angles = []
        for param in self.debug_params:
            try:
                angle = self._p.readUserDebugParameter(param)
                target_angles.append(angle)
            except Exception:
                break
        if len(target_angles) == len(self.joints):
            self.set_joint_position(target_angles)

    def _waitsleep(self, t, sleep=False):
        if t is not None:
            iters = int(t*self._freq)
            for _ in range(iters):
                self._p.stepSimulation()
                self._total_steps += 1
                if sleep:
                    while time.time() - self._last_time < self._timestep:
                        time.sleep(0.0001)
                    self._last_time = time.time()
