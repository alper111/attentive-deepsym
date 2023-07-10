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
        self.agent = manipulators.Manipulator(p=self._p, path="ur10e/ur10e.urdf", position=[0., 0., 0.4], ik_idx=10)
        base_constraint = self._p.createConstraint(parentBodyUniqueId=self.env_dict["base"], parentLinkIndex=0,
                                                   childBodyUniqueId=self.agent.id, childLinkIndex=-1,
                                                   jointType=self._p.JOINT_FIXED, jointAxis=(0, 0, 0),
                                                   parentFramePosition=(0, 0, 0),
                                                   childFramePosition=(0.0, 0.0, -0.2),
                                                   childFrameOrientation=(0, 0, 0, 1))
        self._p.changeConstraint(base_constraint, maxForce=10000)
        # force grippers to act in sync
        mimic_constraint = self._p.createConstraint(self.agent.id, 8, self.agent.id, 9,
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


class BlocksWorld_v2(BlocksWorld):
    def __init__(self, **kwargs):
        self.traj_t = 1.5
        self.locs = {
            0: [0.8, -0.18, 0.41],
            1: [0.8, 0.0, 0.41],
            2: [0.8, 0.18, 0.41],
            3: [0.5, -0.3, 0.41],
            4: [0.5, 0.0, 0.41],
            5: [0.5, 0.3, 0.41]
        }
        self.sizes = [[0.025, 0.025, 0.05], [0.025, 0.2, 0.025]]
        super(BlocksWorld_v2, self).__init__(**kwargs)

    def init_objects(self):
        if self.min_objects == 1:
            self.num_objects = np.random.choice([1, 2, 3], p=[0.1, 0.2, 0.7])
        else:
            self.num_objects = 3
        obj_types = [1, 0, 0]
        R = np.random.permutation(3)
        self.current_obj_locs = [[] for _ in self.locs]
        i = 0
        obj_ids = []
        while i < self.num_objects:
            loc_idx = np.random.randint(3, 6)
            size_idx = obj_types[R[i]]
            if len(self.current_obj_locs[loc_idx]) > 0:
                continue

            position = self.locs[loc_idx][:2] + [0.6]
            size = self.sizes[size_idx]
            self.current_obj_locs[loc_idx].append(0)
            obj_ids.append(utils.create_object(p=self._p, obj_type=self._p.GEOM_BOX,
                                               size=size, position=position, rotation=[0, 0, 0],
                                               mass=0.1, color="random"))
            i += 1
        for i, o_id in enumerate(sorted(obj_ids)):
            self.obj_dict[i] = o_id

    def step(self, from_loc, to_loc, sleep=False):
        target_quat = self._p.getQuaternionFromEuler([np.pi, 0, np.pi/2])
        from_pos = self.locs[from_loc]
        from_top_pos = from_pos[:2] + [0.8]
        to_pos = self.locs[to_loc]
        to_top_pos = to_pos[:2] + [0.8]

        before_pose = self.state_obj_poses()
        self.agent.set_cartesian_position(from_top_pos, orientation=target_quat, t=self.traj_t, traj=True, sleep=sleep)
        self.agent.move_in_cartesian(from_pos, orientation=target_quat, t=self.traj_t, sleep=sleep)
        self.agent.close_gripper(self.traj_t, sleep=sleep)
        self.agent.move_in_cartesian(from_top_pos, orientation=target_quat, t=self.traj_t, sleep=sleep)
        self.agent.move_in_cartesian(to_top_pos, orientation=target_quat, t=self.traj_t, sleep=sleep, ignore_force=True)
        self.agent._waitsleep(0.3, sleep=sleep)
        self.agent.move_in_cartesian(to_pos, orientation=target_quat, t=self.traj_t, sleep=sleep)
        self.agent._waitsleep(0.5, sleep=sleep)
        self.agent.open_gripper(self.traj_t, sleep=sleep)
        self.agent.move_in_cartesian(to_top_pos, orientation=target_quat, t=self.traj_t, sleep=sleep)
        self.init_agent_pose(t=1.0, sleep=sleep)
        after_pose = self.state_obj_poses()
        effect = after_pose - before_pose
        if len(self.current_obj_locs[from_loc]) > 0:
            self.current_obj_locs[from_loc].pop()
        self.current_obj_locs[to_loc].append(0)
        return effect

    def sample_random_action(self):
        if np.random.rand() < 0.2:
            # there might be actions that does not pick any objects
            from_idx = np.random.randint(6)
        else:
            from_idx = np.random.choice([i for i in range(len(self.current_obj_locs))
                                         if len(self.current_obj_locs[i]) > 0])
        to_idx = np.random.randint(6)
        return (from_idx, to_idx)


class BlocksWorld_v4(BlocksWorld):
    def __init__(self, x_area=0.5, y_area=1.0, **kwargs):
        self.traj_t = 1.5

        self.x_init = 0.5
        self.y_init = -0.5
        self.x_final = self.x_init + x_area
        self.y_final = self.y_init + y_area

        ds = 0.075
        self.ds = ds

        # TODO: ADD GRAPH
        self.obj_types = {}
        if 'min_objects' not in kwargs:
            kwargs["min_objects"] = 8
        if 'max_objects' not in kwargs:
            kwargs["max_objects"] = 13

        single_size = 0.025
        self.sizes = [[single_size, single_size, 0.05],
                      [single_size, 5*single_size, 0.025],
                      [5*single_size, 0.025, single_size]]
        self.debug_items = []

        super(BlocksWorld_v4, self).__init__(**kwargs)
        self.previous_action = 0

    def create_object_from_db(self, state_row):
        obj_type = state_row[-1]
        position = [state_row[0], state_row[1], state_row[2]]
        rotation = [state_row[3], state_row[4], state_row[5]]
        if obj_type == 0:
            return
        size = self.sizes[0].copy()
        if obj_type == 4:
            size = self.sizes[1]
        elif obj_type == 5:
            size = self.sizes[2]
        elif obj_type == 3:
            size[2] = self.sizes[2][2]
        # obj type is never 0
        if obj_type == 0:
            o_id = utils.create_object(p=self._p, obj_type=self._p.GEOM_SPHERE,
                                       size=size, position=position, rotation=rotation,
                                       mass=0.1, color="random")
        elif obj_type == 1:
            o_id = (utils.create_object(p=self._p, obj_type=self._p.GEOM_BOX,
                                        size=size, position=position, rotation=rotation,
                                        mass=0.1, color="random"))
        elif obj_type == 2:
            o_id = (utils.create_object(p=self._p, obj_type=self._p.GEOM_CYLINDER,
                                        size=size, position=position, rotation=rotation,
                                        mass=0.1, color="random"))
        elif obj_type == 3:
            o_id = (utils.create_object(p=self._p, obj_type=self._p.GEOM_BOX,
                                        size=size, position=position, rotation=rotation,
                                        mass=0.1, color="random"))
        elif obj_type == 4:
            o_id = (utils.create_object(p=self._p, obj_type=self._p.GEOM_BOX,
                                        size=size, position=position, rotation=rotation,
                                        mass=0.1, color="random"))
        elif obj_type == 5:
            o_id = (utils.create_object(p=self._p, obj_type=self._p.GEOM_BOX,
                                        size=size, position=position, rotation=rotation,
                                        mass=0.1, color="random"))
        self.obj_types[o_id] = obj_type
        return o_id

    def create_object(self, obj_type, x, y, z=0.5):
        """
        Add an object ot the world, without collusions
        return -1 if it is not possible
        return object id if possible
        """

        position = [x, y, z]

        size = self.sizes[0].copy()
        if obj_type == 4:
            size = self.sizes[1]
        elif obj_type == 5:
            size = self.sizes[2]
        elif obj_type == 1:
            size[2] = self.sizes[2][2]
        elif obj_type == 2:
            size[1] = 0.05

        o_id = -1
        # obj type is never 0
        if obj_type == 0:
            o_id = utils.create_object(p=self._p, obj_type=self._p.GEOM_SPHERE,
                                       size=size, position=position, rotation=[0, 0, 0],
                                       mass=0.1, color="random")
        elif obj_type == 1:
            o_id = (utils.create_object(p=self._p, obj_type=self._p.GEOM_BOX,
                                        size=size, position=position, rotation=[0, 0, 0],
                                        mass=0.1, color="random"))
        elif obj_type == 2:
            o_id = (utils.create_object(p=self._p, obj_type=self._p.GEOM_CYLINDER,
                                        size=size, position=position, rotation=[0, 0, 0],
                                        mass=0.1, color="random"))
        elif obj_type == 3:
            o_id = (utils.create_object(p=self._p, obj_type=self._p.GEOM_BOX,
                                        size=size, position=position, rotation=[0, 0, 0],
                                        mass=0.1, color="random"))
        elif obj_type == 4:
            o_id = (utils.create_object(p=self._p, obj_type=self._p.GEOM_BOX,
                                        size=size, position=position, rotation=[0, 0, 0],
                                        mass=0.1, color="random"))
        elif obj_type == 5:
            o_id = (utils.create_object(p=self._p, obj_type=self._p.GEOM_BOX,
                                        size=size, position=position, rotation=[0, 0, np.pi],
                                        mass=0.1, color="random"))
        self.obj_types[o_id] = obj_type
        return o_id

    def init_objects(self):
        '''
        obj_tpes index:
            0 : sphere
            1 : box
            2 : cylinder
            3 : tall
            4 : long box
            5 : long box rotated
        '''
        self.obj_buffer = []
        self.obj_types = {}
        obj_ids = []
        self.num_objects = np.random.randint(self.min_objects, self.max_objects+1)
        obj_types = np.random.choice([1, 4], size=(self.num_objects,), replace=True)

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
                if np.any(distances < 0.15):
                    trials += 1
                    if trials > 10:
                        z = 0.57
                    else:
                        continue

            trials = 0
            obj_id = self.create_object(obj_type, x, y, z)
            if obj_id == -1:
                continue

            positions.append(pos)
            obj_ids.append(obj_id)
            self._p.addUserDebugText(str(i), [0, 0, 0.1], [0, 0, 0], 1, 0, parentObjectUniqueId=obj_id)
            i += 1

        self.cluster_centers = []
        for i in range(np.random.randint(1, 3)):
            self.cluster_centers.append(np.random.randint(0, self.num_objects))
        for i, o_id in enumerate(sorted(obj_ids)):
            self.obj_dict[i] = o_id

    def remove_grid(self):
        for line in self.debug_items:
            self._p.removeUserDebugItem(line)
        self.debug_items = []

    def print_grid(self, location):
        line_x_color = [0, 0, 1]
        line_y_color = [1, 0, 1]
        x = location[0]
        y = location[1]
        z = location[2]
        for i in [-1, 0, 1]:
            id1 = self._p.addUserDebugLine([x + self.ds * i, y + self.ds, z],
                                           [x + self.ds * i, y - self.ds, z],
                                           lifeTime=0,
                                           lineWidth=0.25,
                                           lineColorRGB=line_x_color)
            id2 = self._p.addUserDebugLine([x + self.ds, y + self.ds * i, z],
                                           [x - self.ds, y + self.ds * i, z],
                                           lifeTime=0,
                                           lineWidth=0.125,
                                           lineColorRGB=line_y_color)
            self.debug_items.append(id1)
            self.debug_items.append(id2)

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
        if sleep:
            self.remove_grid()
            self.print_grid(obj1_loc)
            self.print_grid(obj2_loc)
        obj1_loc[0] += dx1 * self.ds
        obj2_loc[0] += dx2 * self.ds
        obj1_loc[1] += dy1 * self.ds
        obj2_loc[1] += dy2 * self.ds
        # obj1_loc[2] -= 0.01
        # obj2_loc[2] -= 0.01

        from_top_pos = obj1_loc.copy()
        from_top_pos[2] = 0.9

        to_top_pos = obj2_loc.copy()
        to_top_pos[2] = 0.9
        state1, types = self.state()

        self.agent.set_cartesian_position(from_top_pos, orientation=quat1, t=self.traj_t, traj=True, sleep=sleep)
        self.agent.move_in_cartesian(obj1_loc, orientation=quat1, t=self.traj_t, sleep=sleep)
        if get_images:
            images.append(utils.get_image(self._p, eye_position=eye_position, target_position=target_position,
                                          up_vector=up_vector, height=256, width=256)[0])
        self.agent.close_gripper(self.traj_t, sleep=sleep)
        self.agent.move_in_cartesian(from_top_pos, orientation=quat1, t=self.traj_t, sleep=sleep)
        state2, _ = self.state()
        # self.agent.move_in_cartesian(from_top_pos, orientation=quat2, t=self.traj_t, sleep=sleep)
        self.agent.move_in_cartesian(to_top_pos, orientation=quat2, t=self.traj_t, sleep=sleep, ignore_force=True)
        self.agent._waitsleep(0.3, sleep=sleep)
        if get_images:
            images.append(utils.get_image(self._p, eye_position=eye_position, target_position=target_position,
                                          up_vector=up_vector, height=256, width=256)[0])
        state3, _ = self.state()
        self.agent.move_in_cartesian(obj2_loc, orientation=quat2, t=self.traj_t, sleep=sleep)
        self.agent._waitsleep(0.5, sleep=sleep)
        self.agent.open_gripper()
        self.agent.move_in_cartesian(to_top_pos, orientation=quat2, t=self.traj_t, sleep=sleep)
        if get_images:
            images.append(utils.get_image(self._p, eye_position=eye_position, target_position=target_position,
                                          up_vector=up_vector, height=256, width=256)[0])
        state4, _ = self.state()
        effect = np.concatenate([state2 - state1, state4 - state3], axis=1)
        self.init_agent_pose(t=1.0, sleep=sleep)
        if get_images:
            return state1, effect, types, images

        return state1, effect, types

    def _state_obj_poses_and_types(self):
        N_obj = len(self.obj_dict)
        pose = np.zeros((N_obj, 9), dtype=np.float32)
        obj_types = np.zeros((N_obj), dtype=np.int8)
        for i in range(N_obj):
            position, quaternion = self._p.getBasePositionAndOrientation(self.obj_dict[i])
            pose[i][:3] = position
            euler_angles = self._p.getEulerFromQuaternion(quaternion)
            pose[i][3:] = [np.cos(euler_angles[0]), np.sin(euler_angles[0]),
                           np.cos(euler_angles[1]), np.sin(euler_angles[1]),
                           np.cos(euler_angles[2]), np.sin(euler_angles[2])]
            obj_types[i] = self.obj_types[self.obj_dict[i]]

        return pose, obj_types

    def state(self):
        return self._state_obj_poses_and_types()

    def full_random_action(self):
        obj1_id, obj2_id = np.random.choice(list(self.obj_dict.keys()), size=(2,), replace=False)
        if self.obj_types[self.obj_dict[obj1_id]] == 4:
            dy1 = np.random.choice([-1, 0, 1])
        else:
            dy1 = 0
        dy2 = np.random.choice([-1, 0, 1])
        return [obj1_id, obj2_id, 0, dy1, 0, dy2, 1, 1]

    def sample_random_action(self, p1=None, p2=None):
        obj1 = np.random.randint(self.num_objects)
        obj2 = np.random.choice(self.num_objects)

        while obj1 in self.cluster_centers:
            obj1 = np.random.randint(self.num_objects)
        dx1, dy1, dx2, dy2 = 0, 0, 0, 0
        dxdy_pairs = [[0, 0], [0, 1], [1, 0],
                      [-1, 0], [0, -1], [-1, 1],
                      [-1, -1], [1, 1], [1, -1]]
        if p1 is None:
            p1 = [0.6] + [0.075] * 4 + [0.025] * 4
        if p2 is None:
            p2 = [0.4] + [0.125] * 4 + [0.025] * 4
        dxdy1 = np.random.choice(np.arange(len(dxdy_pairs)), p=p1)
        dxdy2 = np.random.choice(np.arange(len(dxdy_pairs)), p=p2)

        [dx1, dy1] = dxdy_pairs[dxdy1]
        [dx2, dy2] = dxdy_pairs[dxdy2]
        dx1 = 0
        dx2 = 0
        # rot_before, rot_after = np.random.randint(0, 2, (2))
        rot_before, rot_after = (1, 1)
        return [obj1, obj2, dx1, dy1, dx2, dy2, rot_before, rot_after]

    def sample_3_objects_moving_together(self):
        long_objects = []
        smalls = []

        for i in self.obj_dict.keys():
            if self.obj_types[self.obj_dict[i]] == 4:
                long_objects.append(i)
            else:
                smalls.append(i)

        long_object = random.choice(long_objects)
        small1, small2 = random.choices(smalls, k=2)
        while small1 == small2:
            small1 = random.choice(smalls)

        act = self.sample_random_action()
        if long_object == 4:
            self.obj_buffer.append([small1, long_object, 0, 0, 1, 0, 1, 1])
            self.obj_buffer.append([small2, long_object, 0, 0, -1, 0, 1, 1])
            act[6] = 0
        else:
            self.obj_buffer.append([small1, long_object, 0, 0, 0, 1, 1, 1])
            self.obj_buffer.append([small2, long_object, 0, 0, 0, -1, 1, 1])
        act[6] = 1
        act[2] = 0
        act[3] = 0
        act[0] = long_object
        self.obj_buffer.append(act)
        return self.obj_buffer

    def sample_mistarget(self):
        obj1 = np.random.randint(self.num_objects)
        obj2 = np.random.choice(self.num_objects)
        obj3 = np.random.choice(self.num_objects)
        while obj1 == obj2:
            obj2 = np.random.choice(self.num_objects)
        while obj2 == obj3 or obj1 == obj3:
            obj3 = np.random.choice(self.num_objects)
        self.obj_buffer.append(
                [obj1, obj2, 0, 0, 0, 0, 1, 1]
            )
        self.obj_buffer.append(
                self.sample_random_action()
            )
        self.obj_buffer.append(
                [obj2, obj3, 0, 0, 0, 0, 1, 1]
            )
        return self.obj_buffer

    def sample_both(self):
        # TODO: introduce clustering
        long_objects = []
        smalls = []

        for i in self.obj_dict.keys():
            if self.obj_types[self.obj_dict[i]] == 4:
                long_objects.append(i)
            else:
                smalls.append(i)

        long_object = long_objects.pop(0)
        smalls += long_objects
        self.obj_buffer.append([smalls[0], long_object, 0, 0, 0, 1, 1, 1])
        self.obj_buffer.append([smalls[1], long_object, 0, 0, 0, -1, 1, 1])
        self.obj_buffer.append([smalls[2], smalls[3], 0, 0, 0, 0, 1, 1])
        return self.obj_buffer

    def sample_proximity(self):
        while len(self.obj_buffer) < 8:
            act = self.sample_random_action(p1=[1] + [0] * 4 + [0] * 4, p2=[0] + [1/8] * 8)
            if act[0] != act[1]:
                self.obj_buffer.append(act)
        return self.obj_buffer

    def sample_long_rotation(self):
        long_objects = []
        smalls = []
        for i in self.obj_dict.keys():
            if self.obj_types[self.obj_dict[i]] == 4:
                long_objects.append(i)
            else:
                smalls.append(i)

        long = long_objects.pop(0)
        small = smalls[0]
        locs = [-1, 0, 1]
        locs.remove(np.random.randint(-1, 2))
        loc1, loc2 = locs

        self.obj_buffer.append([small, long, 0, 0, 0, loc1, 1, 1])
        self.obj_buffer.append([long, long, 0, loc1, 0, loc2, 1, 1])
        self.obj_buffer.append([long, np.random.randint(0, self.num_objects), 0,
                                loc1, 0, np.random.randint(-1, 2), 1, 1])
        return self.obj_buffer

    def sample_ungrappable(self):
        tall = 0
        small = 0
        for i in self.obj_dict.keys():
            if self.obj_types[self.obj_dict[i]] == 3:
                tall = i
            elif self.obj_types[self.obj_dict[i]] < 3:
                small = i
        axis = random.random()
        if axis < 0.5:
            dx = np.random.choice([1, -1])
            self.obj_buffer.append([tall, small, 0, 0, dx, 0, 0, 0])
            self.obj_buffer.append([small, np.random.randint(0, self.num_objects), 0, 0, 0, 0, 1, 0])
        else:
            dy = np.random.choice([1, -1])
            self.obj_buffer.append([tall, small, 0, 0, 0, dy, 0, 0])
            self.obj_buffer.append([small, np.random.randint(0, self.num_objects), 0, 0, 0, 0, 0, 0])
        return self.obj_buffer


class PushEnv(GenericEnv):
    def __init__(self, gui=0, seed=None):
        super(PushEnv, self).__init__(gui=gui, seed=seed)

    def reset(self, seed=None):
        super(PushEnv, self).reset(seed=seed)

        self.obj_dict = {}
        self.init_agent_pose(t=1)
        self.init_objects()
        self._step(40)
        self.agent.close_gripper(1, sleep=True)

    def reset_objects(self):
        for key in self.obj_dict:
            obj_id = self.obj_dict[key]
            self._p.removeBody(obj_id)
        self.obj_dict = {}
        self.init_objects()
        self._step(240)

    def init_objects(self):
        obj_type = np.random.choice([self._p.GEOM_BOX, self._p.GEOM_SPHERE, self._p.GEOM_CYLINDER], p=[0.6, 0.1, 0.3])
        position = [0.8, 0.0, 0.6]
        rotation = [0, 0, 0]
        if self._p.GEOM_CYLINDER:
            r = np.random.uniform(0.05, 0.15)
            h = np.random.uniform(0.05, 0.15)
            size = [r, h]
        else:
            r = np.random.uniform(0.05, 0.15)
            size = [r, r, r]
        size = np.random.uniform(0.015, 0.035, (3,)).tolist()

        self.obj_dict[0] = utils.create_object(p=self._p, obj_type=obj_type, size=size, position=position,
                                               rotation=rotation, color="random", mass=0.1)

    def state(self):
        rgb, depth, seg = utils.get_image(p=self._p, eye_position=[1.5, 0.0, 1.5], target_position=[0.9, 0., 0.4],
                                          up_vector=[0, 0, 1], height=256, width=256)
        return rgb[:, :, :3], depth, seg

    def step(self, action, sleep=False):
        traj_time = 1

        if action == 0:
            self.agent.set_cartesian_position([0.8, -0.15, 0.75],
                                              self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=0.5, sleep=sleep)
            self.agent.set_cartesian_position([0.8, -0.15, 0.42],
                                              self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=0.5, sleep=sleep)
            self.agent.move_in_cartesian([0.8, 0.15, 0.42],
                                         self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=traj_time, sleep=sleep)
            self.init_agent_pose(t=0.25, sleep=sleep)
        elif action == 1:
            self.agent.set_cartesian_position([0.95, 0.0, 0.75],
                                              self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=0.5, sleep=sleep)
            self.agent.set_cartesian_position([0.95, 0.0, 0.42],
                                              self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=0.5, sleep=sleep)
            self.agent.move_in_cartesian([0.65, 0.0, 0.42],
                                         self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=traj_time, sleep=sleep)
            self.init_agent_pose(t=0.25, sleep=sleep)
        elif action == 2:
            self.agent.set_cartesian_position([0.8, 0.15, 0.75],
                                              self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=0.5, sleep=sleep)
            self.agent.set_cartesian_position([0.8, 0.15, 0.42],
                                              self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=0.5, sleep=sleep)
            self.agent.move_in_cartesian([0.8, -0.15, 0.42],
                                         self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=traj_time, sleep=sleep)
            self.init_agent_pose(t=0.25, sleep=sleep)
        elif action == 3:
            self.agent.set_cartesian_position([0.65, 0.0, 0.75],
                                              self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=0.5, sleep=sleep)
            self.agent.set_cartesian_position([0.65, 0.0, 0.42],
                                              self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=0.5, sleep=sleep)
            self.agent.move_in_cartesian([0.95, 0.0, 0.42],
                                         self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=traj_time, sleep=sleep)
            self.init_agent_pose(t=0.25, sleep=sleep)
