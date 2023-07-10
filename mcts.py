import time
from copy import deepcopy
import multiprocessing as mp

import numpy as np
import torch

import utils


class Tree:
    def __init__(self, root_state, max_nodes=100000, forward_fn=None, min_forward_count=10):
        self.max_nodes = max_nodes
        self.min_forward_count = min_forward_count
        self.node = []  # [state_id0, state_id1, ...]
        self.id_to_state = []  # [state0, state1, ...]
        self.state_to_id = {}  # {state0: state_id0, state1: state_id1, ...}
        self.transition_table = {}  # {(state_id, action_id): (next_state_id, count)}
        self.forward_fn = forward_fn  # state = forward_fn(state, action)
        self.children = []  # [{action_id0: [node_id0, node_id1, ...], action_id1: [node_id2, node_id3, ...]}, ...]

        self.id_to_action = []  # [action0, action1, ...]
        self.action_to_id = {}  # {action0: action_id0, action1: action_id1, ...}

        self.parent = -1 * np.ones(max_nodes, dtype=np.int64)
        self.parent_action = -1 * np.ones(max_nodes, dtype=np.int64)
        self.count = np.zeros(max_nodes, dtype=np.int64)
        self.reward = np.zeros(max_nodes, dtype=np.float)
        self.is_terminal = np.zeros(max_nodes, dtype=np.bool)

        self.node.append(0)
        self.id_to_state.append(root_state)
        self.children.append({})
        self.is_terminal[0] = root_state.is_terminal()

    def search(self, iter_limit, time_limit, default_depth_limit=1, default_batch_size=1):
        i = 0
        start = time.time()
        end = time.time()
        time_elapsed = end - start
        n_nodes, _ = self._tree_stats()
        start_node_count = n_nodes
        while (i < iter_limit) and (time_elapsed < time_limit) and (n_nodes < self.max_nodes):
            node_id = self._tree_policy()
            reward = 0.0

            # sequential
            for _ in range(default_batch_size):
                reward += self._default_policy(node_id, default_depth_limit)
            reward /= default_batch_size

            self._backup(node_id, reward)

            i += 1
            end = time.time()
            time_elapsed = end - start
            n_nodes, max_depth = self._tree_stats()
            if i % 100 == 0:
                print(f"Tree depth={max_depth}, node count={n_nodes}, "
                      f"node/sec={(n_nodes - start_node_count)/time_elapsed:.2f}, "
                      f"best yield={self.best_yield():.2f}")

    def plan(self):
        terminal_nodes = (self.is_terminal == True)
        if terminal_nodes.sum() == 0:
            return []
        best_node_id = np.argmax(self.reward[terminal_nodes])
        actions = []
        while best_node_id != 0:
            action_id = self.parent_action[best_node_id]
            actions.insert(0, self.id_to_action[action_id])
            best_node_id = self.parent[best_node_id]
        return actions

    def all_plans(self):
        terminal_nodes = (self.is_terminal == True)
        if terminal_nodes.sum() == 0:
            return []
        best_node_ids = np.argsort(self.reward[terminal_nodes])[::-1]
        plans = []
        for best_node_id in best_node_ids:
            actions = []
            while best_node_id != 0:
                action_id = self.parent_action[best_node_id]
                actions.insert(0, self.id_to_action[action_id])
                best_node_id = self.parent[best_node_id]
            plans.append(actions)
        return plans

    def best_yield(self):
        terminal_nodes = (self.reward > 0)
        if terminal_nodes.sum() == 0:
            return 0.0
        best_node_id = np.argmax(self.reward[terminal_nodes])
        return self.reward[best_node_id] / self.count[best_node_id]

    def _tree_policy(self):
        node_id = 0
        while not self.is_terminal[node_id]:
            state_id = self.node[node_id]
            state = self.id_to_state[state_id]
            action_ids = self._get_available_actions(state)
            children = self.children[node_id]
            n_children = len(children)
            if n_children != len(action_ids):
                return self._expand(node_id)
            else:
                uct = np.zeros(n_children)
                child_action_ids = list(children.keys())
                for i, action_id in enumerate(child_action_ids):
                    outcomes = children[action_id]
                    outcome_n = self.count[outcomes]
                    outcome_r = self.reward[outcomes]
                    outcome_uct = outcome_r/outcome_n + np.sqrt((2*np.log(self.count[node_id])) / outcome_n)
                    p = outcome_n / outcome_n.sum()
                    uct[i] = (p * outcome_uct).sum()
                best_action_id = child_action_ids[np.argmax(uct)]
                if np.random.rand() < 1:
                    next_state_id = self._get_next_state_id(state_id, best_action_id)
                    outcome_nodes = children[best_action_id]
                    outcome_state_ids = [self.node[node_id] for node_id in outcome_nodes]
                    try:
                        # if next state is already observed
                        idx = outcome_state_ids.index(next_state_id)
                        node_id = outcome_nodes[idx]
                    except ValueError:
                        # a new outcome has been observed
                        self.node.append(next_state_id)
                        new_node_id = len(self.node) - 1
                        outcome_nodes.append(new_node_id)
                        self.parent[new_node_id] = node_id
                        self.parent_action[new_node_id] = best_action_id
                        self.children.append({})
                        node_id = new_node_id
                else:
                    outcome_n = self.count[children[best_action_id]]
                    p = outcome_n / outcome_n.sum()
                    node_id = np.random.choice(children[best_action_id], p=p)
        return node_id

    def _expand(self, node_id):
        state_id = self.node[node_id]
        action_ids = self._get_available_actions(self.id_to_state[state_id])
        children = self.children[node_id]
        n_children = len(children)
        if n_children != len(action_ids):
            for action_id in action_ids:
                if action_id not in children:
                    next_state_id = self._get_next_state_id(state_id, action_id)
                    self.node.append(next_state_id)
                    next_node_id = len(self.node) - 1
                    self.parent[next_node_id] = node_id
                    self.parent_action[next_node_id] = action_id
                    children[action_id] = [next_node_id]
                    self.children.append({})
                    self.is_terminal[next_node_id] = self.id_to_state[next_state_id].is_terminal()
                    return next_node_id
        else:
            raise ValueError("The node is already fully expanded.")

    def _default_policy(self, node_id, depth_limit):
        depth = 0
        state_id = self.node[node_id]
        while (not self.id_to_state[state_id].is_terminal()) and (depth < depth_limit):
            available_action_ids = self._get_available_actions(self.id_to_state[state_id])
            action_id = np.random.choice(available_action_ids)
            state_id = self._get_next_state_id(state_id, action_id)
            depth += 1
        return self.id_to_state[state_id].reward()

    def _backup(self, node_id, reward):
        while node_id != -1:
            self.count[node_id] += 1
            self.reward[node_id] += reward
            node_id = self.parent[node_id]

    def _get_next_state_id(self, state_id, action_id):
        # mutates states and transition_table
        if (state_id, action_id) in self.transition_table:
            outcomes = self.transition_table[(state_id, action_id)]
            outcome_counts = np.array([outcome[1] for outcome in outcomes])
            total_count = outcome_counts.sum()
            if total_count > self.min_forward_count:
                p = outcome_counts / total_count
                outcome_idx = np.random.choice(len(outcomes), p=p)
                next_state_id = outcomes[outcome_idx][0]
                return next_state_id

        state = self.id_to_state[state_id]
        action = self.id_to_action[action_id]
        next_state = self.forward_fn(state, action)
        in_array, array_idx = utils.in_array(next_state, self.id_to_state)
        if in_array:
            next_state_id = array_idx
        else:
            next_state_id = len(self.id_to_state)
            self.id_to_state.append(next_state)

        if (state_id, action_id) in self.transition_table:
            outcomes = self.transition_table[(state_id, action_id)]
            for outcome in outcomes:
                if outcome[0] == next_state_id:
                    self.transition_table[(state_id, action_id)] = [(next_state_id, outcome[1] + 1)]
        else:
            self.transition_table[(state_id, action_id)] = [(next_state_id, 1)]

        return next_state_id

    def _get_available_actions(self, state):
        # mutates id_to_action and action_to_id
        available_actions = state.get_available_actions()
        actions = []
        for action in available_actions:
            if action not in self.action_to_id:
                self.action_to_id[action] = len(self.id_to_action)
                self.id_to_action.append(action)
            actions.append(self.action_to_id[action])
        return actions

    def _tree_stats(self):
        n_nodes = len(self.node)
        max_depth = 0
        for node_id in range(n_nodes):
            depth = 0
            while node_id != -1:
                node_id = self.parent[node_id]
                depth += 1
            if depth > max_depth:
                max_depth = depth
        return n_nodes, max_depth


class MCTSNode:
    total_nodes = 1

    def __init__(self, node_id, parent, state, forward_fn):
        self.node_id = node_id
        self.parent = parent
        self.state = state
        self.count = 0
        self.reward = 0
        self.is_terminal = state.is_terminal()
        self._forward_fn = forward_fn
        if self.is_terminal:
            self.actions = None
            self.children = None
        else:
            self.actions = state.get_available_actions()
            self.children = {}

    def _default_policy_wrapper(self, node, default_depth_limit):
        return node._default_policy(default_depth_limit)

    def run(self, iter_limit, time_limit, default_depth_limit=10, default_batch_size=1, n_proc=1):
        i = 0
        start = time.time()
        end = time.time()
        time_elapsed = end - start
        start_node_count, _ = self._tree_stats()
        while (i < iter_limit) and (time_elapsed < time_limit):
            v_arr = self._tree_policy(n=n_proc)
            v_arr = [(v, default_depth_limit) for v in v_arr]
            with mp.get_context('spawn').Pool(processes=n_proc) as pool:
                rewards = pool.starmap(self._default_policy_wrapper, v_arr)

            for (v, _), r in zip(v_arr, rewards):
                v._backup(r)

            i += 1
            end = time.time()
            time_elapsed = end - start
            if i > 1:
                node_count, depth = self._tree_stats()
                print(f"Tree depth={depth}, node count={node_count}, "
                      f"node/sec={(node_count-start_node_count)/time_elapsed:.2f}, "
                      f"best yield={self.best_yield():.5f}")

        return self.children_yield()

    def best_actions(self, n=1):
        children_uct = self.children_uct()
        # sort children_uct dict by value
        children_uct = sorted(children_uct.items(), key=lambda item: item[1], reverse=True)
        best_actions = []
        for i in range(n):
            best_actions.append(children_uct[i][0])
        return best_actions

    def best_yield(self):
        best_yield = -1
        if self.children is None:
            return self.reward/self.count

        for action in self.children:
            for outcome in self.children[action]:
                yield_ = outcome.best_yield()
                if yield_ > best_yield:
                    best_yield = yield_
        if best_yield == -1:
            best_yield = self.reward/self.count
        return best_yield

    def best_action_for_plan(self):
        best_action = None
        best_yield = -np.inf
        children_yield = self.children_yield()
        for action in children_yield:
            if children_yield[action] > best_yield:
                best_action = action
                best_yield = children_yield[action]
        return best_action

    def children_uct(self):
        if not self.is_terminal:
            scores = {}
            for action in self.children:
                next_states = self.children[action]
                probs = []
                bounds = []
                # there may be stochastic outcomes for the same action
                for outcome in next_states:
                    probs.append(outcome.count)
                    bounds.append(outcome.UCT())
                probs = np.array(probs)
                bounds = np.array(bounds)
                probs = probs/probs.sum()
                scores[action] = (probs * bounds).sum()
            return scores
        else:
            return None

    def children_yield(self):
        if not self.is_terminal:
            scores = {}
            for action in self.children:
                next_states = self.children[action]
                probs = []
                yields = []
                # there may be stochastic outcomes for the same action
                for outcome in next_states:
                    probs.append(outcome.count)
                    yields.append(outcome.reward/outcome.count)
                probs = np.array(probs)
                yields = np.array(yields)
                probs = probs / probs.sum()
                scores[action] = (probs * yields).sum()
            return scores
        else:
            return None

    def UCT(self):
        if self.parent is None:
            return None
        else:
            N = self.parent.count
            score = self.reward/self.count + np.sqrt((2*np.log(N)) / self.count)
            return score

    def plan(self, sample=False, best_yield=False):
        assert not (sample and best_yield)
        if self.is_terminal:
            return self.state, "", [], 1.0
        action = self.best_action_for_plan()
        if action is None:
            print("Plan might not be successful.")
            return self.state, "", [(action, 0)], 1.0
        elif len(self.children[action]) == 1:
            child_state, child_plan_txt, child_plan, child_prob = self.children[action][0].plan()
            return child_state, "_".join(filter(None, [action, child_plan_txt])), \
                [(action, 0)]+child_plan, child_prob
        else:
            probs = []
            yields = []
            for out in self.children[action]:
                probs.append(out.count)
                yields.append(out.reward/out.count)
            probs = np.array(probs)
            yields = np.array(yields)
            probs = probs / probs.sum()
            if sample:
                sampled_idx = np.random.choice(len(probs), p=probs)
            elif best_yield:
                sampled_idx = np.argmax(yields)
            else:
                sampled_idx = np.argmax(probs)
            p = probs[sampled_idx]
            child_state, child_plan_txt, child_plan, child_prob = self.children[action][sampled_idx].plan()
            return child_state, "_".join(filter(None, [action, child_plan_txt])), \
                [(action, sampled_idx)]+child_plan, p*child_prob

    def _expand(self, n=1):
        untried_moves = []
        for action in self.actions:
            if action not in self.children:
                untried_moves.append(action)
        if n > len(untried_moves):
            n = len(untried_moves)
        indices = np.random.choice(range(len(untried_moves)), size=n, replace=False)
        actions = [untried_moves[idx] for idx in indices]
        # ACT HERE #
        children = []
        for j, action in enumerate(actions):
            next_state = self._forward_fn(self.state, action)
            self.children[action] = [MCTSNode(node_id=MCTSNode.total_nodes,
                                              parent=self,
                                              state=next_state,
                                              forward_fn=self._forward_fn)]
            MCTSNode.total_nodes += 1
            children.append(self.children[action][0])
        ############
        return children

    def _tree_policy(self, n=1):
        if self.is_terminal:
            return [self]
        # if there is an unexpanded node, first expand it.
        if len(self.children) != len(self.actions):
            return self._expand(n=n)
        # else choose the best child by UCT
        else:
            # have to change here
            actions = self.best_actions(n=n)
            selected = []
            for action in actions:
                if np.random.rand() < 0:  # TODO: this is only for ILP, set it to 0.05 later
                    next_state = self._forward_fn(self.state, action)
                    children_states = list(map(lambda x: x.state, self.children[action]))
                    result, out_idx = utils.in_array(next_state, children_states)
                    if not result:
                        self.children[action].append(MCTSNode(node_id=MCTSNode.total_nodes,
                                                              parent=self,
                                                              state=next_state,
                                                              forward_fn=self._forward_fn,
                                                              n_nodes=self.n_nodes+1))
                        MCTSNode.total_nodes += 1
                        selected.append(self.children[action][-1]._tree_policy()[0])
                        # return self.children[action][-1]._tree_policy()
                    else:
                        selected.append(self.children[action][out_idx]._tree_policy()[0])
                        # return self.children[action][out_idx]._tree_policy()
                else:
                    probs = []
                    for child in self.children[action]:
                        probs.append(child.count)
                    probs = np.array(probs)
                    probs = probs / probs.sum()
                    random_idx = np.random.choice(len(self.children[action]), p=probs)
                    v = self.children[action][random_idx]._tree_policy()
                    selected.append(v[0])
                    # return self.children[action][random_idx]._tree_policy()
            return selected

    def _default_policy(self, depth_limit):
        if (not self.is_terminal) and (depth_limit > 0):
            random_action_idx = np.random.choice(len(self.actions))
            random_action = self.actions[random_action_idx]
            # ACT HERE #
            next_state = self._forward_fn(self.state, random_action)
            v = MCTSNode(node_id=-1, parent=None, state=next_state, forward_fn=self._forward_fn)
            ############
            return v._default_policy(depth_limit-1)
        else:
            return self.state.reward()

    def _backup(self, reward):
        if self.parent is not None:
            self.count += 1
            self.reward += reward
            self.parent._backup(reward)
        else:
            self.count += 1
            self.reward += reward

    def __repr__(self):
        state = str(self.state)
        children = []
        if not self.is_terminal:
            children_scores = []
            children_uct = self.children_uct()
            for action in children_uct:
                children_scores.append(children_uct[action])
            for action in self.children:
                outcomes = list(map(lambda x: str(x.node_id), self.children[action]))
                outcomes = "[" + ", ".join(outcomes) + "]"
                children.append(outcomes)
        string = "Node id: " + str(self.node_id) + "\n"
        if self.parent:
            string += "Parent: " + str(self.parent.node_id) + "\n"
        else:
            string += "Parent: None\n"
        if not self.is_terminal:
            string += "Children: [" + ", ".join(children) + "]\n"
            string += "Children UCT: [" + ", ".join([str(s) for s in children_scores]) + "]\n"
        string += "State:\n" + state + "\n"
        string += "Reward: " + str(self.reward) + "\n"
        string += "Count: " + str(self.count) + "\n"
        string += "Terminal: " + str(self.is_terminal)
        return string

    def _tree_stats(self):
        if self.is_terminal:
            return 1, 0

        children_depths = []
        total_nodes = 1
        for action in self.children:
            c = self.children[action]
            if len(c) == 1:
                nodes, depth = c[0]._tree_stats()
                children_depths.append(depth)
                total_nodes += nodes
            else:
                gchild_depths = []
                for c_i in c:
                    nodes, depth = c_i._tree_stats()
                    gchild_depths.append(depth)
                    total_nodes += nodes
                children_depths.append(max(gchild_depths)+1)
        if len(children_depths) == 0:
            return total_nodes, 0

        return total_nodes, max(children_depths)+1


class MCTSState:
    def __init__(self, goal):
        pass

    def reward(self):
        pass

    def get_available_actions(self):
        pass

    def is_terminal(self):
        pass

    def __repr__(self):
        pass

    def is_equal(self, other):
        pass


class MCTSForward:
    def __init__(self):
        pass

    def forward(self, state, action):
        pass


class SymbolicForwardModel(MCTSForward):
    def __init__(self, model):
        self.model = model

    def __call__(self, state, action):
        obj_symbol, rel_symbol = utils.to_tensor_state(state.state)
        action = torch.tensor([int(a_i) for a_i in action.split(",")])
        action_placeholder = torch.zeros(obj_symbol.shape[0], 8)  # (grasp_or_release, dx_loc, dy_loc, rot)
        action_placeholder[action[0], :4] = torch.tensor([1, action[1], action[2], 1], dtype=torch.float)
        action_placeholder[action[3], 4:] = torch.tensor([1, action[4], action[5], 1], dtype=torch.float)
        z_cat = torch.cat([obj_symbol, action_placeholder], dim=-1)
        with torch.no_grad():
            obj_symbol_next, rel_symbol_next = self.model(z_cat.unsqueeze(0), rel_symbol.unsqueeze(0))
        obj_symbol_next = obj_symbol_next.sigmoid().bernoulli()[0]
        rel_symbol_next = rel_symbol_next.sigmoid().bernoulli()[0]
        str_state = utils.to_str_state(obj_symbol_next, rel_symbol_next, torch.ones(obj_symbol_next.shape[0]))
        return SymbolicState(state=deepcopy(str_state), goal=deepcopy(state.goal))


class SymbolicState(MCTSState):
    available_actions = []
    n_obj = 5
    for i in range(n_obj):
        for iy in range(-1, 2):
            for j in range(n_obj):
                for jy in range(-1, 2):
                    available_actions.append(f"{i},0,{iy},{j},0,{jy}")

    def __init__(self, state, goal):
        self.state = state
        self.goal = goal

    def reward(self):
        reward = int(self.is_terminal())
        return reward

    def get_available_actions(self):
        return SymbolicState.available_actions

    def is_terminal(self):
        return self.state == self.goal

    def is_equal(self, other):
        return self.state == other.state

    def __repr__(self):
        return self.state


class SubsymbolicState(MCTSState):
    threshold = 0.025
    available_actions = []
    n_obj = 5
    for i in range(n_obj):
        for iy in range(-1, 2):
            for j in range(n_obj):
                for jy in range(-1, 2):
                    available_actions.append(f"{i},0,{iy},{j},0,{jy}")

    def __init__(self, state, goal):
        self.state = state
        self.goal = goal

    def reward(self):
        # diff = self.goal_diff()
        # reward = min(SubsymbolicState.threshold / diff, 1)
        return int(self.is_terminal())

    def goal_diff(self):
        diff = (self.state[:, :3] - self.goal[:, :3]).abs().mean(dim=0).sum()
        return diff

    def get_available_actions(self):
        return SubsymbolicState.available_actions

    def is_terminal(self):
        diff = self.goal_diff()
        return diff < SubsymbolicState.threshold

    def is_equal(self, other):
        diff = (self.state[:, :3] - other.state[:, :3]).abs().mean(dim=0).sum()
        return diff < SubsymbolicState.threshold

    def __repr__(self):
        return self.state.__repr__()


class SubsymbolicForwardModel(MCTSForward):
    def __init__(self, model):
        self.model = model

    def __call__(self, state, action):
        n_objs = state.state.shape[0]
        mask = torch.ones(1, n_objs)
        action = torch.tensor([int(a_i) for a_i in action.split(",")])
        action_placeholder = torch.zeros(state.state.shape[0], 8)  # (grasp_or_release, dx_loc, dy_loc, rot)
        action_placeholder[action[0], :4] = torch.tensor([1, action[1], action[2], 1], dtype=torch.float)
        action_placeholder[action[3], 4:] = torch.tensor([1, action[4], action[5], 1], dtype=torch.float)
        inp = {
            "state": state.state.unsqueeze(0),
            "action": action_placeholder.unsqueeze(0),
            "pad_mask": mask
        }
        with torch.no_grad():
            _, _, e = self.model.forward(inp, eval_mode=False)
            delta_pos = state.state[action[3]] - state.state[action[0]]
            dx, dy = delta_pos[0], delta_pos[1]
            dx += (-action[1] * 0.075) + (action[4] * 0.075)
            dy += (-action[2] * 0.075) + (action[5] * 0.075)
            next_state = state.state.clone()
        next_state[:, :3] = next_state[:, :3] + e[0, :, :3] + e[0, :, 3:]
        # print(dx, dy, e[0, :, :3], e[0, :, 3:])
        for i in range(n_objs):
            # if the object is lifted, it should be moved
            if e[0, i, 2] > 0.1:
                next_state[i, 0] += dx
                next_state[i, 1] += dy

        return SubsymbolicState(state=next_state, goal=state.goal)
