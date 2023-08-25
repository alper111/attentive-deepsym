import pybullet
import pybullet_data
import numpy as np
import random

import utils
import manipulators


class GenericEnv:
    def __init__(self, gui=0, seed=None):
        self._p = utils.connect(gui)
        self.gui = gui
        self.reset(seed=seed)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self._p.resetSimulation()
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setGravity(0, 0, -9.807)
        self._p.loadURDF("plane.urdf")

        self.env_dict = utils.create_tabletop(self._p)
        self.agent = manipulators.Manipulator(p=self._p, path="ur10e/ur10e.urdf", position=[0., 0., 0.4], ik_idx=30)
        base_constraint = self._p.createConstraint(parentBodyUniqueId=self.env_dict["base"], parentLinkIndex=0,
                                                   childBodyUniqueId=self.agent.id, childLinkIndex=-1,
                                                   jointType=self._p.JOINT_FIXED, jointAxis=(0, 0, 0),
                                                   parentFramePosition=(0, 0, 0),
                                                   childFramePosition=(0.0, 0.0, -0.2),
                                                   childFrameOrientation=(0, 0, 0, 1))
        self._p.changeConstraint(base_constraint, maxForce=10000)
        # force grippers to act in sync
        mimic_constraint = self._p.createConstraint(self.agent.id, 28, self.agent.id, 29,
                                                    jointType=self._p.JOINT_GEAR,
                                                    jointAxis=[1, 0, 0],
                                                    parentFramePosition=[0, 0, 0],
                                                    childFramePosition=[0, 0, 0])
        self._p.changeConstraint(mimic_constraint, gearRatio=-1, erp=0.1, maxForce=50)

    def init_agent_pose(self, t=None, sleep=False, traj=False):
        angles = [-0.294, -1.950, 2.141, -2.062, -1.572, 1.277]
        self.agent.set_joint_position(angles, t=t, sleep=sleep, traj=traj)

    def state_obj_poses(self):
        N_obj = len(self.obj_dict)
        pose = np.zeros((N_obj, 7), dtype=np.float32)
        for i in range(N_obj):
            position, quaternion = self._p.getBasePositionAndOrientation(self.obj_dict[i])
            pose[i][:3] = position
            pose[i][3:] = quaternion
        return pose

    def _step(self, count=1):
        for _ in range(count):
            self._p.stepSimulation()

    def __del__(self):
        self._p.disconnect()


class BlocksWorld(GenericEnv):
    def __init__(self, gui=0, seed=None, min_objects=2, max_objects=5):
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.num_objects = None
        super(BlocksWorld, self).__init__(gui=gui, seed=seed)

    def reset(self, seed=None):
        super(BlocksWorld, self).reset(seed=seed)

        self.obj_dict = {}
        self.init_agent_pose(t=1)
        self.init_objects()
        self._step(40)
        self.agent.open_gripper(1, sleep=True)

    def delete_objects(self):
        for key in self.obj_dict:
            obj_id = self.obj_dict[key]
            self._p.removeBody(obj_id)
        self.obj_dict = {}

    def reset_objects(self):
        self.delete_objects()
        self.init_objects()
        self._step(240)

    def reset_object_poses(self):
        for key in self.obj_dict:
            x = np.random.uniform(0.5, 1.0)
            y = np.random.uniform(-0.4, 0.4)
            z = np.random.uniform(0.6, 0.65)
            quat = pybullet.getQuaternionFromEuler(np.random.uniform(0, 90, (3,)).tolist())

            self._p.resetBasePositionAndOrientation(self.obj_dict[key], [x, y, z], quat)
        self._step(240)

    def init_objects(self):
        self.num_objects = np.random.randint(self.min_objects, self.max_objects+1)
        for i in range(self.num_objects):
            obj_type = np.random.choice([self._p.GEOM_BOX], p=[1])
            x = np.random.uniform(0.5, 1.0)
            y = np.random.uniform(-0.4, 0.4)
            z = np.random.uniform(0.6, 0.65)
            size = np.random.uniform(0.015, 0.035, (3,)).tolist()
            rotation = np.random.uniform(0, 90, (3,)).tolist()
            # if obj_type == self._p.GEOM_CAPSULE:
            #     rotation = [0, 0, 0]

            if obj_type == self._p.GEOM_BOX:
                # color = [1.0, 0.0, 0.0, 1.0]
                if np.random.rand() < 0.4:
                    size = [np.random.uniform(0., 0.2), np.random.uniform(0.01, 0.015),
                            np.random.uniform(0.015, 0.025)]
                    # color = [0.0, 1.0, 1.0, 1.0]
            self.obj_dict[i] = utils.create_object(p=self._p, obj_type=obj_type, size=size, position=[x, y, z],
                                                   rotation=rotation, color="random", mass=0.1)

    def state(self):
        rgb, depth, seg = utils.get_image(p=self._p, eye_position=[1.2, 0.0, 1.6], target_position=[0.8, 0., 0.4],
                                          up_vector=[0, 0, 1], height=256, width=256)
        return rgb[:, :, :3], depth, seg

    def step(self, from_obj_id, to_obj_id, sleep=False):
        from_pos, from_quat = self._p.getBasePositionAndOrientation(from_obj_id)
        to_pos, to_quat = self._p.getBasePositionAndOrientation(to_obj_id)
        to_pos = to_pos[:2] + (0.75,)
        traj_time = 0.5
        self.agent.set_cartesian_position(from_pos[:2]+(0.75,),
                                          orientation=self._p.getQuaternionFromEuler([np.pi, 0, 0]),
                                          t=traj_time,
                                          traj=True,
                                          sleep=sleep)
        self.agent.set_cartesian_position(from_pos,
                                          orientation=self._p.getQuaternionFromEuler([np.pi, 0, 0]),
                                          t=traj_time,
                                          traj=True,
                                          sleep=sleep)
        self.agent.close_gripper(traj_time, sleep=sleep)
        self.agent.set_cartesian_position(from_pos[:2]+(0.75,),
                                          orientation=self._p.getQuaternionFromEuler([np.pi, 0, 0]),
                                          t=traj_time,
                                          traj=True,
                                          sleep=sleep)
        self.agent.set_cartesian_position(to_pos,
                                          orientation=self._p.getQuaternionFromEuler([np.pi, 0, 0]),
                                          t=traj_time,
                                          traj=True,
                                          sleep=sleep)
        # self.agent._waitsleep(0.5, sleep=sleep)
        before_pose = self.state_obj_poses()
        self.agent.open_gripper(traj_time, sleep=sleep)
        self.init_agent_pose(t=1.0, sleep=sleep)
        after_pose = self.state_obj_poses()
        effect = after_pose - before_pose
        return effect


class BlocksWorld_v4(BlocksWorld):
    def __init__(self, x_area=0.5, y_area=1.0, **kwargs):
        self.traj_t = 1.5
        self.x_init = 0.5
        self.y_init = -0.5
        self.x_final = self.x_init + x_area
        self.y_final = self.y_init + y_area

        ds = 0.075
        self.ds = ds

        self.obj_types = {}

        self.sizes = [[0.025, 0.025, 0.025],
                      [0.025, 0.125, 0.025]]
        self.colors = [[1.0, 0.0, 0.0, 1.0],
                       [0.0, 1.0, 0.0, 1.0],
                       [0.0, 0.0, 1.0, 1.0],
                       [1.0, 1.0, 0.0, 1.0],
                       [1.0, 0.0, 1.0, 1.0],
                       [0.0, 1.0, 1.0, 1.0]]
        self.obj_enc = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]])
        self.obj_dict = {}
        super(BlocksWorld_v4, self).__init__(**kwargs)

    def init_objects(self):
        self.obj_types = {}
        obj_ids = []
        self.num_objects = np.random.randint(self.min_objects, self.max_objects+1)
        obj_types = np.random.choice([0, 1], size=(self.num_objects,), replace=True)
        colors = np.random.choice([0, 1, 2, 3, 4, 5], size=(self.num_objects,), replace=False)
        colors = [self.colors[c] for c in colors]

        i = 0
        positions = []
        trials = 0
        total_trials = 0
        while i < self.num_objects:
            total_trials += 1
            if total_trials > 100:
                print("Could not place all objects, retrying...")
                return self.init_objects()

            obj_type = obj_types[i]
            x = np.random.uniform(self.x_init, self.x_final)
            y = np.random.uniform(self.y_init, self.y_final)
            z = 0.43
            pos = np.array([x, y])
            positions = []
            if len(positions) > 0:
                distances = np.linalg.norm(np.stack(positions) - pos, axis=-1)
                if np.any(distances < 0.20):
                    trials += 1
                    if trials > 10:
                        z = 0.57
                    else:
                        continue

            trials = 0
            o_id = utils.create_object(p=self._p, obj_type=self._p.GEOM_BOX,
                                       size=self.sizes[obj_type], position=[x, y, z], rotation=[0, 0, 0],
                                       mass=0.1, color=colors[i])
            self.obj_types[o_id] = obj_type

            positions.append(pos)
            obj_ids.append(o_id)
            self._p.addUserDebugText(str(i), [0, 0, 0.1], [0, 0, 0], 1, 0, parentObjectUniqueId=o_id)
            i += 1
        for i, o_id in enumerate(sorted(obj_ids)):
            self.obj_dict[i] = o_id

    def step(self, obj1_id, obj2_id, dx1, dy1, dx2, dy2, rotated_grasp,
             rotated_release, sleep=False, get_images=False):
        eye_position = [1.75, 0.0, 2.0]
        target_position = [1.0, 0.0, 0.4]
        up_vector = [0, 0, 1]
        images = []
        if get_images:
            images.append(utils.get_image(self._p, eye_position=eye_position, target_position=target_position,
                                          up_vector=up_vector, height=256, width=256)[0])

        obj1_loc, _ = self._p.getBasePositionAndOrientation(self.obj_dict[obj1_id])
        obj2_loc, _ = self._p.getBasePositionAndOrientation(self.obj_dict[obj2_id])
        grasp_angle1 = [np.pi, 0, 0]
        grasp_angle2 = [np.pi, 0, 0]
        if rotated_grasp:
            grasp_angle1[2] = np.pi/2
        if rotated_release:
            grasp_angle2[2] = np.pi/2
        quat1 = self._p.getQuaternionFromEuler(grasp_angle1)
        quat2 = self._p.getQuaternionFromEuler(grasp_angle2)
        obj1_loc = list(obj1_loc)
        obj2_loc = list(obj2_loc)
        obj1_loc[0] += dx1 * self.ds
        obj2_loc[0] += dx2 * self.ds
        obj1_loc[1] += dy1 * self.ds
        obj2_loc[1] += dy2 * self.ds
        from_top_pos = obj1_loc.copy()
        from_top_pos[2] = 0.9
        to_top_pos = obj2_loc.copy()
        to_top_pos[2] = 0.9

        state1 = self.state()  # before grasp
        # start grasping
        self.agent.set_cartesian_position(from_top_pos, orientation=quat1, t=self.traj_t, traj=True, sleep=sleep)
        self.agent.move_in_cartesian(obj1_loc, orientation=quat1, t=self.traj_t, sleep=sleep)
        if get_images:
            images.append(utils.get_image(self._p, eye_position=eye_position, target_position=target_position,
                                          up_vector=up_vector, height=256, width=256)[0])
        self.agent.close_gripper(self.traj_t, sleep=sleep)

        # move upwards
        self.agent.move_in_cartesian(from_top_pos, orientation=quat1, t=self.traj_t, sleep=sleep)
        state2 = self.state()

        # move to target
        self.agent.move_in_cartesian(to_top_pos, orientation=quat2, t=self.traj_t, sleep=sleep, ignore_force=True)
        self.agent._waitsleep(0.3, sleep=sleep)
        if get_images:
            images.append(utils.get_image(self._p, eye_position=eye_position, target_position=target_position,
                                          up_vector=up_vector, height=256, width=256)[0])
        state3 = self.state()

        # release
        self.agent.move_in_cartesian(obj2_loc, orientation=quat2, t=self.traj_t, sleep=sleep)
        self.agent._waitsleep(0.5, sleep=sleep)
        self.agent.open_gripper()

        # move upwards
        self.agent.move_in_cartesian(to_top_pos, orientation=quat2, t=self.traj_t, sleep=sleep)
        if get_images:
            images.append(utils.get_image(self._p, eye_position=eye_position, target_position=target_position,
                                          up_vector=up_vector, height=256, width=256)[0])
        self.init_agent_pose(t=1.0, sleep=sleep)
        self.agent._waitsleep(2)
        state4 = self.state()
        delta_pos1 = state2[:, :3] - state1[:, :3]
        delta_quat1 = np.stack([self._p.getDifferenceQuaternion(q2, q1) for q1, q2 in zip(state1[:, 3:7], state2[:, 3:7])])
        delta_pos2 = state4[:, :3] - state3[:, :3]
        delta_quat2 = np.stack([self._p.getDifferenceQuaternion(q2, q1) for q1, q2 in zip(state3[:, 3:7], state4[:, 3:7])])
        effect = np.concatenate([delta_pos1, delta_quat1, delta_pos2, delta_quat2], axis=-1)
        if get_images:
            return state1, effect, images

        return state1, effect

    def state(self):
        poses = self.state_obj_poses()
        obj_types = np.array([self.obj_types[self.obj_dict[i]] for i in range(len(self.obj_dict))])
        state = np.concatenate([poses, self.obj_enc[obj_types]], axis=1)
        return state

    def full_random_action(self):
        obj1_id, obj2_id = np.random.choice(list(self.obj_dict.keys()), size=(2,), replace=True)
        dy1 = np.random.choice([-1, 0, 1])
        dy2 = np.random.choice([-1, 0, 1])
        return [obj1_id, obj2_id, 0, dy1, 0, dy2, 1, 1]
