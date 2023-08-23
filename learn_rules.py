import argparse
import multiprocessing as mp
import pickle
import os
from functools import reduce
from copy import deepcopy

import torch
import numpy as np
import lightning.pytorch as pl

import dataset
from models import load_ckpt


class Node:
    def __init__(self, left, right, object_bindings, action_bindings, relation_bindings, counts, gating) -> None:
        self.left = left
        self.right = right
        self.object_bindings = object_bindings
        self.action_bindings = action_bindings
        self.relation_bindings = relation_bindings
        self.counts = counts
        self.gating = gating
        self.named_effects = {}

    def __repr__(self) -> str:
        return f"Node({self.object_bindings}, {self.action_bindings}, {self.relation_bindings}, {self.gating.sum()})"


def create_effect_classes(loader, given_effect_to_class=None):
    if given_effect_to_class is None:
        effect_to_class = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}}
    class_idx = [0, 0, 0, 0, 0]  # 1 object symbol, 4 relational symbols
    effects = []
    changed_indices = []
    for _, (obj_pre, rel_pre, _, obj_post, rel_post, mask) in enumerate(loader):
        obj_pre = obj_pre[0, mask[0]]
        obj_post = obj_post[0, mask[0]]
        obj_diff_idx = torch.where(obj_pre != obj_post)[0]
        obj_diff_idx = torch.unique(obj_diff_idx)
        obj_effects = []
        obj_indices = []
        for idx in obj_diff_idx:
            obj_effects.append(tuple(obj_post[idx].int().tolist()))
            obj_indices.append(idx.item())
        # sort obj_effects together with obj_indices
        if len(obj_effects) > 0:
            obj_effects, obj_indices = zip(*sorted(zip(obj_effects, obj_indices)))
        else:
            obj_effects = ()
            obj_indices = ()

        mm = (mask.T.float() @ mask.float()).bool()
        c = mask.sum()
        rel_pre = rel_pre[0, :, mm].reshape(4, c, c)
        rel_post = rel_post[0, :, mm].reshape(4, c, c)
        rel_diffs = torch.where(rel_pre != rel_post)
        rel_effects = [[], [], [], []]  # for four relations
        rel_indices = [[], [], [], []]
        for rel, obj1, obj2 in zip(*rel_diffs):
            rel_value = rel_post[rel, obj1, obj2]
            tup = (rel_value.int().item(),)
            rel_effects[rel.item()].append(tup)
            rel_indices[rel.item()].append((obj1.item(), obj2.item()))

        # sort rel_effects together with rel_indices
        for j in range(4):
            if len(rel_effects[j]) == 0:
                rel_effects[j] = ()
                rel_indices[j] = ()
                continue
            rel_effects[j], rel_indices[j] = zip(*sorted(zip(rel_effects[j], rel_indices[j])))
        rel_indices = tuple(rel_indices)

        key = (obj_effects,) + tuple(rel_effects)
        effect_class = []
        for j, k_i in enumerate(key):
            if given_effect_to_class is not None:
                if k_i not in given_effect_to_class[j]:
                    effect_class.append(-1)
                else:
                    effect_class.append(given_effect_to_class[j][k_i])
            elif k_i not in effect_to_class[j]:
                effect_to_class[j][k_i] = class_idx[j]
                class_idx[j] += 1
                effect_class.append(effect_to_class[j][k_i])
            else:
                effect_class.append(effect_to_class[j][k_i])
        effects.append(tuple(effect_class))

        changed_indices.append((obj_indices, *rel_indices))

    if given_effect_to_class is not None:
        return effects, changed_indices
    return effects, changed_indices, effect_to_class


def get_top_classes(sorted_effect_counts, dataset_size, total_perc=None, min_samples=None):
    assert total_perc is not None or min_samples is not None
    total_count = 0
    selected_keys = []
    for key in sorted_effect_counts:
        count = sorted_effect_counts[key]
        total_count += count
        if (min_samples is not None) and (count < min_samples):
            break
        if (total_perc is not None) and ((total_count / dataset_size) > total_perc):
            break
        selected_keys.append(key)
    return selected_keys, total_count/dataset_size


def get_effect_counts(effects, gating):
    effect_counts = {}
    for i, e in enumerate(effects):
        if gating[i]:
            if e not in effect_counts:
                effect_counts[e] = 0
            effect_counts[e] += 1
    effect_counts = dict(sorted(effect_counts.items(), key=lambda x: x[1], reverse=True))
    return effect_counts


def filter_effect_classes(effects, selected_classes):
    filtered_effects = []
    for e in effects:
        if e in selected_classes:
            filtered_effects.append(e)
        else:
            filtered_effects.append(-1)
    return filtered_effects


def matrix_to_tuple(matrix):
    return tuple([tuple(row) for row in matrix])


def preprocess_data(o_i, r_i, a, o_f, r_f, m):
    o_i = o_i[0, m[0]].int()
    if o_f is not None:
        o_f = o_f[0, m[0]].int()
    c = m.sum()
    mm = (m.T.float() @ m.float()).bool()
    r_i = r_i[0, :, mm].reshape(4, c, c).int()
    if r_f is not None:
        r_f = r_f[0, :, mm].reshape(4, c, c).int()
    a = a[0, m[0]].int()
    return o_i, r_i, a, o_f, r_f


def is_satisfied(sample, object_bindings, action_bindings, relation_bindings):
    o_i, r_i, a, _, _ = preprocess_data(*sample)

    # get possible object indices
    obj_exists = True
    obj_possible_indices = {}
    for name in object_bindings:
        shared_possible_indices = []
        for (val, direction) in object_bindings[name]:
            # val, direction = object_bindings[name]
            if direction == 0:
                indices = torch.where((o_i == val).all(dim=1))[0]
            else:
                indices = torch.where((o_i != val).any(dim=1))[0]

            if len(indices) > 0:
                shared_possible_indices.append(indices)
                # obj_possible_indices[name] = indices
            else:
                obj_exists = False
                break
        if obj_exists:
            obj_possible_indices[name] = torch.tensor(
                reduce(np.intersect1d, (x.numpy() for x in shared_possible_indices)),
                dtype=torch.long)
        else:
            break

    # get possible action indices
    act_exists = True
    act_possible_indices = {}
    for name in action_bindings:
        shared_possible_indices = []
        for (val, direction) in action_bindings[name]:
            # val, direction = action_bindings[name]
            if direction == 0:
                indices = torch.where((a == val).all(dim=1))[0]
            else:
                indices = torch.where((a != val).any(dim=1))[0]

            if len(indices) > 0:
                shared_possible_indices.append(indices)
                # act_possible_indices[name] = indices
            else:
                act_exists = False
                break
        if act_exists:
            act_possible_indices[name] = torch.tensor(
                reduce(np.intersect1d, (x.numpy() for x in shared_possible_indices)),
                dtype=torch.long)
        else:
            break

    # constraints
    obj_act_binded = True
    all_names = list(set(list(object_bindings.keys()) + list(action_bindings.keys())))
    filtered_possible_indices = {}
    if len(all_names) == 0:
        obj_act_binded = False

    for name in all_names:
        if name in obj_possible_indices:
            obj_indices = obj_possible_indices[name]
        else:
            obj_indices = None

        if name in act_possible_indices:
            act_indices = act_possible_indices[name]
        else:
            act_indices = None

        if obj_indices is None and act_indices is None:
            obj_act_binded = False
            break
        elif obj_indices is None:
            filtered_possible_indices[name] = act_indices
        elif act_indices is None:
            filtered_possible_indices[name] = obj_indices
        else:
            filtered_possible_indices[name] = torch.tensor(np.intersect1d(obj_indices.numpy(),
                                                                          act_indices.numpy()),
                                                           dtype=torch.long)

        if len(filtered_possible_indices[name]) == 0:
            obj_act_binded = False
            break

    possible_bindings = []
    if obj_act_binded:
        tensors = []
        for name in all_names:
            tensors.append(filtered_possible_indices[name])
        bindings = torch.cartesian_prod(*tensors)
        if bindings.ndim == 1:
            bindings = bindings.unsqueeze(1)
        num_vars = len(all_names)
        for binding in bindings:
            if torch.unique(binding).shape[0] == num_vars:
                possible_bindings.append({all_names[i]: binding[i] for i in range(num_vars)})
        if len(possible_bindings) == 0:
            obj_act_binded = False

    rel_filtered_bindings = []
    for binding in possible_bindings:
        binding_valid = True
        for (rel_idx, name1, name2) in relation_bindings:
            val, direction = relation_bindings[(rel_idx, name1, name2)]
            name1_idx = binding[name1]
            name2_idx = binding[name2]
            if direction == 0:
                if r_i[rel_idx, name1_idx, name2_idx] != val:
                    binding_valid = False
                    break
            else:
                if r_i[rel_idx, name1_idx, name2_idx] == val:
                    binding_valid = False
                    break
        if binding_valid:
            rel_filtered_bindings.append(binding)
    rel_exists = len(rel_filtered_bindings) > 0
    satisfied = obj_exists and act_exists and obj_act_binded and rel_exists

    return satisfied, rel_filtered_bindings


def check_rule(object_bindings, action_bindings, relation_bindings,
               loader, effects, gating):
    left_counts = {}
    right_counts = {}
    left_gating = np.zeros(len(gating), dtype=bool)
    right_gating = np.zeros(len(gating), dtype=bool)
    for i, sample in enumerate(loader):
        if gating[i]:
            satisfied, _ = is_satisfied(sample, object_bindings, action_bindings, relation_bindings)

            if satisfied:
                if effects[i] not in left_counts:
                    left_counts[effects[i]] = 0
                left_counts[effects[i]] += 1
                left_gating[i] = True
            else:
                if effects[i] not in right_counts:
                    right_counts[effects[i]] = 0
                right_counts[effects[i]] += 1
                right_gating[i] = True
    return left_counts, left_gating, right_counts, right_gating


def count_named_effects(node, loader, effect_indices, effect_class_values, rel_idx):
    named_effects = {}
    for i, sample in enumerate(loader):
        satisfied, bindings = is_satisfied(sample, node.object_bindings,
                                           node.action_bindings, node.relation_bindings)
        if satisfied:
            for binding in bindings:
                reverse_mapping = {int(v): k for k, v in binding.items()}
                nm_effect = transform_tuple(effect_indices[i][rel_idx], reverse_mapping)
                # effect_with_value = []
                current_effect_value = effect_class_values[i][rel_idx]
                if (len(nm_effect) > 0) and (len(current_effect_value) == len(nm_effect)):
                    current_rel = []
                    for n_i, name in enumerate(nm_effect):
                        val = current_effect_value[n_i]
                        if isinstance(name, str):
                            current_rel.append((name,) + (val,))
                        else:
                            current_rel.append(name + val)
                elif (len(nm_effect) == 0) and (len(current_effect_value) == 0):
                    current_rel = []
                nm_effect = tuple(current_rel)
                # for r_i, field in enumerate(nm_effect):
                #     if (len(field) > 0) and (len(current_effect_value[r_i]) == len(field)):
                #         current_rel = []
                #         for n_i, name in enumerate(field):
                #             val = current_effect_value[r_i][n_i]
                #             if isinstance(name, str):
                #                 current_rel.append((name,) + (val,))
                #             else:
                #                 current_rel.append(name + val)
                #         effect_with_value.append(tuple(current_rel))
                #     elif (len(field) == 0) and (len(current_effect_value[r_i]) == 0):
                #         effect_with_value.append(())
                # nm_effect = tuple(effect_with_value)
                if nm_effect not in named_effects:
                    named_effects[nm_effect] = 0
                named_effects[nm_effect] += 1
    return named_effects


def populate_named_effects(node, loader, effect_indices, effect_class_values, rel_idx, n_procs=1):
    proc_args = []
    queue = [node]
    while len(queue) > 0:
        node = queue.pop(0)
        proc_args.append((node, loader, effect_indices, effect_class_values, rel_idx))
        if node.left is not None:
            queue.append(node.left)
        if node.right is not None:
            queue.append(node.right)

    with mp.get_context("spawn").Pool(n_procs) as pool:
        results = pool.starmap(count_named_effects, proc_args)

    for (node, _, _, _, _), result in zip(proc_args, results):
        node.named_effects = result


def calculate_entropy(counts):
    probs = {k: counts[k]/np.sum(list(counts.values())) for k in counts}
    entropy = -np.sum([probs[k]*np.log(probs[k]) for k in probs])
    return entropy


def calculate_best_split(node, loader, effects, unique_object_values,
                         unique_action_values, min_samples_split, num_procs=1):
    """
    Calculate the best split for the given node.

    Args:
        node (Node): The node to expand.
        loader (torch.utils.data.DataLoader): The data loader.
        effects (List[int]): The effects.
        effect_indices (List[Tuple[Tuple[int], Tuple[Tuple[int]]]]): The indices of the effects.
        unique_object_values (torch.Tensor): The unique object values.
        unique_action_values (torch.Tensor): The unique action values.
        min_samples_split (int): The minimum number of samples required to split a node.
        num_procs (int): The number of processes to use.

    Returns:
        Tuple[float, Node]: The entropy and the best node.
    """
    left_node = None
    right_node = None
    best_impurity = 1e10
    if node.gating.sum() < min_samples_split:
        return best_impurity, left_node, right_node

    obj_var_list = list(node.object_bindings.keys())
    act_var_list = list(node.action_bindings.keys())

    # if a new object variable is needed
    max_obj_idx = max([int(obj_var[3:]) for obj_var in obj_var_list]) if len(obj_var_list) > 0 else -1
    max_act_idx = max([int(act_var[3:]) for act_var in act_var_list]) if len(act_var_list) > 0 else -1
    new_obj_idx = max(max_obj_idx, max_act_idx) + 1
    new_obj_name = "obj" + str(new_obj_idx)

    # process argument list
    proc_args = []
    right_args = []

    # bind a variable in action list to a new object value
    for act_var in act_var_list:
        # continue if the variable already is bound to an object value
        # if act_var in node.object_bindings:
        #     continue

        # bind the variable to each object value
        for obj_val in unique_object_values:
            object_bindings = deepcopy(node.object_bindings)
            if act_var not in object_bindings:
                object_bindings[act_var] = []
            else:
                is_bound_before = False
                # check whether it has been bound to the same value
                for val, _ in object_bindings[act_var]:
                    if (obj_val == val).all():
                        is_bound_before = True
                        continue
                if is_bound_before:
                    continue
            object_bindings[act_var].append((obj_val, 0))
            # object_bindings[act_var] = (obj_val, 0)
            proc_args.append((object_bindings, node.action_bindings, node.relation_bindings,
                              loader, effects, node.gating))
            right_object_bindings = deepcopy(node.object_bindings)
            if act_var not in right_object_bindings:
                right_object_bindings[act_var] = []
            right_object_bindings[act_var].append((obj_val, 1))
            # right_object_bindings[act_var] = (obj_val, 1)
            right_args.append((right_object_bindings, node.action_bindings, node.relation_bindings))

    # bind a new variable to a new object value
    for obj_val in unique_object_values:
        object_bindings = node.object_bindings.copy()
        object_bindings[new_obj_name] = [(obj_val, 0)]
        # object_bindings[new_obj_name] = (obj_val, 0)
        proc_args.append((object_bindings, node.action_bindings, node.relation_bindings,
                          loader, effects, node.gating))
        right_object_bindings = deepcopy(node.object_bindings)
        right_object_bindings[new_obj_name] = [(obj_val, 1)]
        # right_object_bindings[new_obj_name] = (obj_val, 1)
        right_args.append((right_object_bindings, node.action_bindings, node.relation_bindings))

    # bind a variable in object list to a new action value
    for obj_var in obj_var_list:
        # continue if the variable already is bound to an action value
        # if obj_var in node.action_bindings:
        #     continue

        # bind the variable to each action value
        for act_val in unique_action_values:
            action_bindings = deepcopy(node.action_bindings)
            if obj_var not in action_bindings:
                action_bindings[obj_var] = []
            else:
                is_bound_before = False
                # check whether it has been bound to the same value
                for val, _ in action_bindings[obj_var]:
                    if (act_val == val).all():
                        is_bound_before = True
                        continue
                if is_bound_before:
                    continue
            action_bindings[obj_var].append((act_val, 0))
            # action_bindings[obj_var] = (act_val, 0)
            proc_args.append((node.object_bindings, action_bindings, node.relation_bindings,
                              loader, effects, node.gating))
            right_action_bindings = deepcopy(node.action_bindings)
            if obj_var not in right_action_bindings:
                right_action_bindings[obj_var] = []
            right_action_bindings[obj_var].append((act_val, 1))
            # right_action_bindings[obj_var] = (act_val, 1)
            right_args.append((node.object_bindings, right_action_bindings, node.relation_bindings))

    # bind a new variable to a new action value
    for act_val in unique_action_values:
        action_bindings = deepcopy(node.action_bindings)
        action_bindings[new_obj_name] = [(act_val, 0)]
        # action_bindings[new_obj_name] = (act_val, 0)
        proc_args.append((node.object_bindings, action_bindings, node.relation_bindings,
                          loader, effects, node.gating))
        right_action_bindings = deepcopy(node.action_bindings)
        right_action_bindings[new_obj_name] = [(act_val, 1)]
        # right_action_bindings[new_obj_name] = (act_val, 1)
        right_args.append((node.object_bindings, right_action_bindings, node.relation_bindings))

    # bind two variables in either object list or action list to a new relation value
    all_vars = list(set(obj_var_list + act_var_list))
    for v1 in all_vars:
        for v2 in all_vars:
            for rel in [0, 1, 2, 3]:  # TODO: this is hard-coded to four for now
                for val in [0, 1]:
                    key = (rel, v1, v2)

                    # continue if the relation is already bound
                    if key in node.relation_bindings:
                        continue

                    # bind the relation to each value
                    relation_bindings = deepcopy(node.relation_bindings)
                    relation_bindings[key] = (val, 0)
                    proc_args.append((node.object_bindings, node.action_bindings, relation_bindings,
                                      loader, effects, node.gating))
                    right_relation_bindings = deepcopy(node.relation_bindings)
                    right_relation_bindings[key] = (val, 1)
                    right_args.append((node.object_bindings, node.action_bindings, right_relation_bindings))

    with mp.get_context("spawn").Pool(num_procs) as pool:
        results = pool.starmap(check_rule, proc_args)

    for (left_counts, left_gating, right_counts, right_gating), (args), (r_args) in zip(results, proc_args, right_args):
        left_entropy = calculate_entropy(left_counts)
        right_entropy = calculate_entropy(right_counts)
        impurity = (left_entropy * np.sum(left_gating) + right_entropy * np.sum(right_gating)) / node.gating.sum()
        if (1e-8 < impurity < best_impurity) and \
           (np.sum(left_gating) >= min_samples_split) and \
           (np.sum(right_gating) >= min_samples_split):
            left_node = Node(left=None, right=None,
                             object_bindings=deepcopy(args[0]),
                             action_bindings=deepcopy(args[1]),
                             relation_bindings=deepcopy(args[2]),
                             counts=left_counts,
                             gating=left_gating)
            right_node = Node(left=None, right=None,
                              object_bindings=deepcopy(r_args[0]),
                              action_bindings=deepcopy(r_args[1]),
                              relation_bindings=deepcopy(r_args[2]),
                              counts=right_counts,
                              gating=right_gating)
            best_impurity = impurity
    return best_impurity, left_node, right_node


def learn_tree(loader, effects, effect_indices, effect_class_values,
               unique_object_values, unique_action_values, min_samples_split=100,
               num_procs=1):
    """Learn a decision tree from the given dataset.

    Args:
        loader (DataLoader): the dataset loader
        effects (np.ndarray): the effects of the dataset
        effect_indices List[Tuple[Tuple[int], Tuple[Tuple[int]]]]: the indices of the effects
        unique_object_values (torch.Tensor): the unique object values in the dataset
        unique_action_values (torch.Tensor): the unique action values in the dataset
        min_samples_split (int): the minimum number of samples required to split a node

    Returns:
        Node: the root node of the decision tree
    """
    # initialize the root node
    gating = np.ones(len(loader), dtype=bool)
    root_node = Node(left=None, right=None,
                     object_bindings={},
                     action_bindings={},
                     relation_bindings={},
                     counts=get_effect_counts(effects, gating),
                     gating=gating)

    # learn the tree
    queue = [root_node]
    num_nodes = 0
    while len(queue) > 0:
        node = queue.pop(0)
        num_nodes += 1
        _, left_node, right_node = calculate_best_split(node, loader, effects, unique_object_values,
                                                        unique_action_values, min_samples_split, num_procs)
        if left_node is not None:
            print(f"Left node:\n"
                  f"  object bindings={left_node.object_bindings},\n"
                  f"  action bindings={left_node.action_bindings},\n"
                  f"  relation bindings={left_node.relation_bindings},\n"
                  f"  entropy={calculate_entropy(left_node.counts)},\n"
                  f"  count={left_node.gating.sum()},\n"
                  f"Right node:\n"
                  f"  object bindings={right_node.object_bindings},\n"
                  f"  action bindings={right_node.action_bindings},\n"
                  f"  relation bindings={right_node.relation_bindings},\n"
                  f"  entropy={calculate_entropy(right_node.counts)},\n"
                  f"  count={right_node.gating.sum()},\n"
                  f"Num nodes: {num_nodes}")

            node.left = left_node
            node.right = right_node
            queue.append(node.left)
            queue.append(node.right)
            if num_nodes == 1:
                # keep the root node pointer
                root_node = node
        else:
            print(f"Terminal node: \n"
                  f"  object bindings={node.object_bindings},\n"
                  f"  action bindings={node.action_bindings},\n"
                  f"  relation bindings={node.relation_bindings},\n"
                  f"  counts={node.counts},\n"
                  f"  entropy={calculate_entropy(node.counts)},\n"
                  f"Num nodes: {num_nodes}")

    # populate the named effects
    # populate_named_effects(root_node, loader, effect_indices, effect_class_values, num_procs)

    return root_node


def print_tree(node, negatives):
    if node.left is None and node.right is None:
        print("Rule:")
        if len(negatives) > 0:
            print("\t(negations:")
            for neg in negatives:
                print("\t\t(")
                if len(neg[0]) > 0:
                    print("\t\t\t(objects: ", end="")
                    print(" AND ".join([f"{obj}!={tuple(vals.tolist())}" for obj, vals in neg[0].items()]), end="")
                    print(")")
                if len(neg[1]) > 0:
                    print("\t\t\t(actions: ", end="")
                    print(" AND ".join([f"{act}!={tuple(vals.tolist())}" for act, vals in neg[1].items()]), end="")
                    print(")")
                if len(neg[2]) > 0:
                    print("\t\t\t(relations: ", end="")
                    print(" AND ".join([f"rel({rel[0]}, {rel[1]}, {rel[2]})!={vals}" for rel, vals in neg[2].items()]), end="")
                    print(")")
                print("\t\t)")
            print("\t)")

        if len(node.object_bindings) > 0:
            # e.g., obj0=(0, 1, 1, 1)
            print("\t(objects: ", end="")
            print(" AND ".join([f"{obj}={tuple(vals.tolist())}" for obj, vals in node.object_bindings.items()]), end="")
            print(")")
        if len(node.action_bindings) > 0:
            print("\t(actions: ", end="")
            print(" AND ".join([f"{act}={tuple(vals.tolist())}" for act, vals in node.action_bindings.items()]), end="")
            print(")")
        if len(node.relation_bindings) > 0:
            print("\t(relations: ", end="")
            print(" AND ".join([f"rel({rel[0]}, {rel[1]}, {rel[2]})={vals}" for rel, vals in node.relation_bindings.items()]), end="")
            print(")")
        print("\tTHEN")
        print(f"\t{node.counts}")
    else:
        print_tree(node.left, negatives)
        # TODO: this is wrong. a right node does not mean all negating the left node's features
        if len(node.object_bindings) > 0 or len(node.action_bindings) > 0 or len(node.relation_bindings) > 0:
            print_tree(node.right, negatives + [(node.object_bindings, node.action_bindings, node.relation_bindings)])
        else:
            print_tree(node.right, negatives)


def flatten_tuple(nested_tuple):
    flattened = []
    for item in nested_tuple:
        if isinstance(item, tuple):
            flattened.extend(flatten_tuple(item))
        else:
            flattened.append(item)
    return tuple(flattened)


def transform_tuple(nested_tuple, mapping):
    transformed = []
    for item in nested_tuple:
        if isinstance(item, tuple):
            transformed.append(transform_tuple(item, mapping))
        else:
            if item in mapping:
                transformed.append(mapping[item])
            else:
                transformed.append("unk")
    return tuple(transformed)


def collate_preds(preds):
    z_i, r_i, a, z_f, r_f, m = [], [], [], [], [], []
    for pred in preds:
        # zm = (pred["z"] > 0.99) | (pred["z"] < 0.01)
        # zm = zm.all(dim=-1).all(dim=-1)
        # rm = (pred["r"] > 0.99) | (pred["r"] < 0.01)
        # rm = rm.all(dim=-1).all(dim=-1).all(dim=-1)
        # znm = (pred["zn"] > 0.99) | (pred["zn"] < 0.01)
        # znm = znm.all(dim=-1).all(dim=-1)
        # rnm = (pred["rn"] > 0.99) | (pred["rn"] < 0.01)
        # rnm = rnm.all(dim=-1).all(dim=-1).all(dim=-1)
        # high_prob = zm & znm & rm & rnm
        # z_i.append(pred["z"][high_prob].round())
        # r_i.append(pred["r"][high_prob].round())
        # a.append(pred["a"][high_prob].round())
        # z_f.append(pred["zn"][high_prob].round())
        # r_f.append(pred["rn"][high_prob].round())
        # m.append(pred["m"][high_prob])
        z_i.append(pred["z"].bernoulli())
        r_i.append(pred["r"].bernoulli())
        a.append(pred["a"])
        z_f.append(pred["zn"].bernoulli())
        r_f.append(pred["rn"].bernoulli())
        m.append(pred["m"])
    z_i = torch.cat(z_i).cpu().bool()
    r_i = torch.cat(r_i).cpu().bool()
    a = torch.cat(a).cpu().char()
    z_f = torch.cat(z_f).cpu().bool()
    r_f = torch.cat(r_f).cpu().bool()
    m = torch.cat(m).cpu().bool()
    return torch.utils.data.TensorDataset(z_i, r_i, a, z_f, r_f, m)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-n", type=str, required=True, help="Experiment name")
    args.add_argument("-p", type=int, default=1, help="Number of processes")
    args = args.parse_args()

    model, _ = load_ckpt(args.n, tag="best")
    trainer = pl.Trainer(devices=[0])
    cnt_trainloader = torch.utils.data.DataLoader(
        dataset.StateActionEffectDataset("blocks234", "train", obj_relative=False),
        batch_size=1024)
    # cnt_valloader = torch.utils.data.DataLoader(
    #     dataset.StateActionEffectDataset("blocks234", "val", obj_relative=False, n=10000),
    #     batch_size=1024)
    # compute the dataset
    trainset = collate_preds(trainer.predict(model, cnt_trainloader))
    # valset = collate_preds(trainer.predict(model, cnt_valloader))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1)
    train_effects, train_changed_indices, train_effect_classes = create_effect_classes(trainloader)
    train_effects = np.array(train_effects)
    for k in train_effect_classes:
        print(f"{k}: {len(train_effect_classes[k])}")

    train_class_to_effect = {k: {v1: k1 for k1, v1 in v.items()} for k, v in train_effect_classes.items()}
    train_effect_values = []
    for eff in train_effects:
        train_effect_values.append([train_class_to_effect[k][v] for k, v in enumerate(eff)])
    print(f"Example effect class: {train_effects[3]}")
    print(f"Example effect value: {train_effect_values[3]}")
    effect_counts = {}
    selected_classes = {}
    for k in train_effect_classes:
        effect_counts[k] = get_effect_counts(train_effects[:, k], np.ones(len(trainset), dtype=bool))
        print(f"{k}: {effect_counts[k]}")
        selected, perc_covered = get_top_classes(effect_counts[k], len(trainset), total_perc=0.95)
        selected_classes[k] = selected
        print(f"{k}: {len(selected)}, perc: {perc_covered}")
        train_effects[:, k] = filter_effect_classes(train_effects[:, k], selected_classes[k])
        filtered_counts = get_effect_counts(train_effects[:, k], np.ones(len(trainset), dtype=bool))
        print(f"{k}: {filtered_counts}")

    # for sc in selected_classes:
    #     print(f"{sc}: {train_class_to_effect[sc]}, count: {filtered_counts[sc]}")
    # print(f"Total perc: {perc_covered} ({len(selected_classes)} classes)")

    unique_object_values = trainset.tensors[0].int().flatten(0, 1).unique(dim=0)
    unique_action_values = trainset.tensors[2].int().flatten(0, 1).unique(dim=0)

    tree_arr = []
    for k in train_effect_classes:
        print(f"Learning tree for effect {k}")
        root = learn_tree(trainloader, train_effects[:, k], None,
                          None, unique_object_values, unique_action_values,
                          min_samples_split=200, num_procs=args.p)
        tree_arr.append(root)

    save_path = os.path.join("out", args.n)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pickle.dump(tree_arr, open(os.path.join(save_path, "tree.pkl"), "wb"))
    torch.save(train_class_to_effect, os.path.join(save_path, "class_to_effect.pt"))
    torch.save(selected_classes, os.path.join(save_path, "selected_classes.pt"))
