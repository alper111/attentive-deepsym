import argparse
import multiprocessing as mp
import itertools
import pickle


import torch
import wandb
import numpy as np

import dataset


class Node:
    def __init__(self, left, right, object_bindings, action_bindings, relation_bindings, counts, gating) -> None:
        self.left = left
        self.right = right
        self.object_bindings = object_bindings
        self.action_bindings = action_bindings
        self.relation_bindings = relation_bindings
        self.counts = counts
        self.gating = gating

    def __repr__(self) -> str:
        return f"Node({self.object_bindings}, {self.action_bindings}, {self.relation_bindings}, {self.gating.sum()})"


def create_effect_classes(loader, given_effect_to_class=None):
    if given_effect_to_class is None:
        effect_to_class = {}
    class_idx = 0
    effects = []
    changed_indices = []
    for i, (obj_pre, rel_pre, _, obj_post, rel_post, mask) in enumerate(loader):
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
        for i in range(4):
            if len(rel_effects[i]) == 0:
                rel_effects[i] = ()
                rel_indices[i] = ()
                continue
            rel_effects[i], rel_indices[i] = zip(*sorted(zip(rel_effects[i], rel_indices[i])))
        rel_indices = tuple(rel_indices)

        key = (obj_effects,) + tuple(rel_effects)
        if given_effect_to_class is not None:
            if key not in given_effect_to_class:
                effects.append(-1)
            else:
                effects.append(given_effect_to_class[key])
        elif key not in effect_to_class:
            effect_to_class[key] = class_idx
            class_idx += 1
            effects.append(effect_to_class[key])
        else:
            effects.append(effect_to_class[key])
        changed_indices.append((obj_indices, rel_indices))

    if given_effect_to_class is not None:
        return effects, changed_indices
    return effects, changed_indices, effect_to_class


def get_top_classes(sorted_effect_counts, perc, dataset_size):
    total_count = 0
    selected_keys = []
    for key in sorted_effect_counts:
        count = sorted_effect_counts[key]
        total_count += count
        selected_keys.append(key)
        if total_count/dataset_size >= perc:
            break
    return selected_keys


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
    o_f = o_f[0, m[0]].int()
    c = m.sum()
    mm = (m.T.float() @ m.float()).bool()
    r_i = r_i[0, :, mm].reshape(4, c, c).int()
    r_f = r_f[0, :, mm].reshape(4, c, c).int()
    a = a[0, m[0]].int()
    return o_i, r_i, a, o_f, r_f


def is_satisfied(sample, object_bindings, action_bindings, relation_bindings):
    o_i, r_i, a, _, _ = preprocess_data(*sample)

    # get possible object indices
    obj_exists = True
    obj_possible_indices = {}
    for name in object_bindings:
        indices = torch.where((o_i == object_bindings[name]).all(dim=1))[0]
        if len(indices) > 0:
            obj_possible_indices[name] = indices
        else:
            obj_exists = False
            break

    # get possible action indices
    act_exists = True
    act_possible_indices = {}
    for name in action_bindings:
        indices = torch.where((a == action_bindings[name]).all(dim=1))[0]
        if len(indices) > 0:
            act_possible_indices[name] = indices
        else:
            act_exists = False
            break

    # constraints
    obj_act_binded = True
    all_names = list(set(list(object_bindings.keys()) + list(action_bindings.keys())))
    filtered_possible_indices = {}
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
            val = relation_bindings[(rel_idx, name1, name2)]
            name1_idx = binding[name1]
            name2_idx = binding[name2]
            if r_i[rel_idx, name1_idx, name2_idx] != val:
                binding_valid = False
                break
        if binding_valid:
            rel_filtered_bindings.append(binding)
    rel_exists = len(rel_filtered_bindings) > 0
    satisfied = obj_exists and act_exists and obj_act_binded and rel_exists

    return satisfied, rel_filtered_bindings


def check_rule(object_bindings, action_bindings, relation_bindings,
               loader, effects, effect_indices, gating):
    left_counts = {}
    right_counts = {}
    left_gating = np.zeros(len(gating), dtype=bool)
    right_gating = np.zeros(len(gating), dtype=bool)
    effect_bindings = []
    total_binded = 0
    for i, sample in enumerate(loader):
        if gating[i]:
            satisfied, bindings = is_satisfied(sample, object_bindings, action_bindings, relation_bindings)
            flattened_indices = flatten_tuple(effect_indices[i])
            unique_indices = list(set(flattened_indices))
            current_candidates = {}
            for binding in bindings:
                variables = list(binding.keys())
                # TODO: this should work for non-equalities as well
                # think how to do that later
                if sorted(tuple(v.item() for v in binding.values())) != sorted(tuple(unique_indices)):
                    continue

                for perm in itertools.permutations(unique_indices):
                    current_mapping = {p: v for p, v in zip(perm, variables)}
                    computed_indices = tuple([binding[current_mapping[idx]].item() for idx in flattened_indices])
                    similarity = np.sum(np.array(computed_indices) == np.array(flattened_indices)) / len(flattened_indices)
                    if (tuple(perm), tuple(variables)) not in current_candidates:
                        current_candidates[(tuple(perm), tuple(variables))] = []
                    current_candidates[(tuple(perm), tuple(variables))].append(similarity)
            # select the best binding
            if len(current_candidates) > 0:
                best_binding = None
                best_avg_sim = -1
                for binding in current_candidates:
                    avg_sim = np.mean(current_candidates[binding])
                    if avg_sim > best_avg_sim:
                        best_binding = binding
                        best_avg_sim = avg_sim
                best_mapping = {p: v for p, v in zip(best_binding[0], best_binding[1])}
                binding = transform_tuple(effect_indices[i], best_mapping)
                effect_bindings.append(binding)
                total_binded += 1
            else:
                effect_bindings.append(effect_indices[i])

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
    return left_counts, left_gating, right_counts, right_gating, total_binded


def calculate_entropy(counts):
    probs = {k: counts[k]/np.sum(list(counts.values())) for k in counts}
    entropy = -np.sum([probs[k]*np.log(probs[k]) for k in probs])
    return entropy


def calculate_best_split(node, loader, effects, effect_indices, unique_object_values,
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

    # bind a variable in action list to a new object value
    for act_var in act_var_list:
        # continue if the variable already is bound to an object value
        if act_var in node.object_bindings:
            continue

        # bind the variable to each object value
        for obj_val in unique_object_values:
            object_bindings = node.object_bindings.copy()
            object_bindings[act_var] = obj_val
            proc_args.append((object_bindings, node.action_bindings, node.relation_bindings,
                             loader, effects, effect_indices, node.gating))

    # bind a new variable to a new object value
    for obj_val in unique_object_values:
        object_bindings = node.object_bindings.copy()
        object_bindings[new_obj_name] = obj_val
        proc_args.append((object_bindings, node.action_bindings, node.relation_bindings,
                          loader, effects, effect_indices, node.gating))

    # bind a variable in object list to a new action value
    for obj_var in obj_var_list:
        # continue if the variable already is bound to an action value
        if obj_var in node.action_bindings:
            continue

        # bind the variable to each action value
        for act_val in unique_action_values:
            action_bindings = node.action_bindings.copy()
            action_bindings[obj_var] = act_val
            proc_args.append((node.object_bindings, action_bindings, node.relation_bindings,
                              loader, effects, effect_indices, node.gating))

    # bind a new variable to a new action value
    for act_val in unique_action_values:
        action_bindings = node.action_bindings.copy()
        action_bindings[new_obj_name] = act_val
        proc_args.append((node.object_bindings, action_bindings, node.relation_bindings,
                          loader, effects, effect_indices, node.gating))

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
                    relation_bindings = node.relation_bindings.copy()
                    relation_bindings[key] = val
                    proc_args.append((node.object_bindings, node.action_bindings, relation_bindings,
                                      loader, effects, effect_indices, node.gating))

    with mp.get_context("spawn").Pool(num_procs) as pool:
        results = pool.starmap(check_rule, proc_args)

    best_binded = 0
    for (left_counts, left_gating, right_counts, right_gating, total_binded), (args) in zip(results, proc_args):
        left_entropy = calculate_entropy(left_counts)
        right_entropy = calculate_entropy(right_counts)
        impurity = (left_entropy * np.sum(left_gating) + right_entropy * np.sum(right_gating)) / node.gating.sum()
        if (1e-8 < impurity < best_impurity) and \
           (np.sum(left_gating) >= min_samples_split) and \
           (np.sum(right_gating) >= min_samples_split):
            left_node = Node(left=None, right=None,
                             object_bindings=args[0].copy(),
                             action_bindings=args[1].copy(),
                             relation_bindings=args[2].copy(),
                             counts=left_counts,
                             gating=left_gating)
            right_node = Node(left=None, right=None,
                              object_bindings={},
                              action_bindings={},
                              relation_bindings={},
                              counts=right_counts,
                              gating=right_gating)
            best_impurity = impurity
            best_binded = total_binded
    print(best_binded)
    return best_impurity, left_node, right_node


def learn_tree(loader, effects, effect_indices, unique_object_values,
               unique_action_values, min_samples_split=100, num_procs=1):
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
        _, left_node, right_node = calculate_best_split(node, loader, effects, effect_indices,
                                                        unique_object_values, unique_action_values,
                                                        min_samples_split, num_procs)
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
            transformed.append(mapping[item])
    return tuple(transformed)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", type=str, required=True, help="Run id")
    args.add_argument("-p", type=int, default=1, help="Number of processes")
    args = args.parse_args()

    run = wandb.init(entity="colorslab", project="multideepsym", resume="must", id=args.i)

    # load the dataset
    trainset = dataset.load_symbol_dataset("train", run, "cpu")
    valset = dataset.load_symbol_dataset("val", run, "cpu")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1)

    train_effects, train_changed_indices, train_effect_classes = create_effect_classes(trainloader)

    train_class_to_effect = {v: k for k, v in train_effect_classes.items()}
    effect_counts = get_effect_counts(train_effects, np.ones(len(trainset), dtype=bool))
    selected_classes = get_top_classes(effect_counts, 0.95, len(trainset))
    filtered_effects = filter_effect_classes(train_effects, selected_classes)
    filtered_counts = get_effect_counts(filtered_effects, np.ones(len(trainset), dtype=bool))

    unique_object_values = trainset.tensors[0].int().flatten(0, 1).unique(dim=0)
    unique_action_values = trainset.tensors[2].int().flatten(0, 1).unique(dim=0)

    root = learn_tree(trainloader, filtered_effects, train_changed_indices,
                      unique_object_values, unique_action_values,
                      min_samples_split=50, num_procs=args.p)
    pickle.dump(root, open("tree.pkl", "wb"))
