import cProfile
import math
import random

import numpy as np

# from .games.blokus import Game
from numba import njit
from tqdm import tqdm
from treelib import Tree

from util import timeit

FLOAT_MIN = np.finfo(np.float32).min


class Model:
    def __init__(self, policy_logits=780, hidden_state_size=32):
        self.policy_logits = policy_logits
        self.hidden_state_size = hidden_state_size

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def init_infer(self, obs):
        value = np.random.rand(1)
        reward = np.random.rand(1)
        policy_logits = np.random.rand(self.policy_logits)
        hidden_state = np.random.rand(self.hidden_state_size)
        return value, reward, self.softmax(policy_logits), hidden_state

    def infer(self, obs, o=None):
        if obs is not None:
            if len(obs.shape) > 1:
                batch = obs.shape[0]
                value = np.random.rand(batch)
                reward = np.random.rand(batch)
                policy_logits = np.random.rand(batch, self.policy_logits)
                hidden_state = np.random.rand(batch, self.hidden_state_size)
                return value, reward, self.softmax(policy_logits), hidden_state
        value = np.random.rand(1)
        reward = np.random.rand(1)
        policy_logits = np.random.rand(self.policy_logits)
        hidden_state = np.random.rand(self.hidden_state_size)
        return value, reward, self.softmax(policy_logits), hidden_state


@njit
def get_children_ucb(
        parent_node_id,
        children_node_ids,

        id_to_visit_count,
        id_to_prior,
        node_id_to_un_normalized_value,

        max_value,
        min_value,

        pb_c_base,
        pb_c_init,
):
    pb_c = math.log(
        (id_to_visit_count[parent_node_id] + pb_c_base + 1) / pb_c_base
    ) + pb_c_init
    pb_c = pb_c * np.sqrt(id_to_visit_count[parent_node_id]) / ((id_to_visit_count[children_node_ids]) + 1)
    prior_score = pb_c * id_to_prior[children_node_ids]
    value_score = node_id_to_un_normalized_value[children_node_ids]
    value_score = (value_score - min_value) / max_value - min_value
    return prior_score + value_score


@njit
def get_children_value(children_node_ids, id_to_value_sum, id_to_visit_count):
    values = np.zeros(children_node_ids.shape[0])
    indexes_where_positive = (id_to_visit_count[children_node_ids] > 0).nonzero()[0]
    if len(indexes_where_positive) > 0:
        ids_where_positive = children_node_ids[indexes_where_positive]
        values[indexes_where_positive] = id_to_value_sum[ids_where_positive] / id_to_visit_count[ids_where_positive]
    return values


@njit
def get_node_value(node_id, id_to_value_sum, id_to_visit_count):
    if id_to_visit_count[node_id] == 0:
        return 0
    return id_to_value_sum[node_id] / id_to_visit_count[node_id]


@njit
def depth_to_player_turn(depth, invert=False):
    raw_value = (depth // 2) % 2
    if invert:
        if raw_value == 1:
            return 0
        return 1
    return raw_value


@njit
def expand_node_static(
        node_id,
        valid_actions: np.ndarray,
        to_play,
        reward,
        policy_logits: np.ndarray,
        hidden_state: np.ndarray,

        num_of_nodes,
        num_of_expanded_nodes,
        node_id_to_expanded_id,
        node_id_to_player,
        node_id_to_reward,
        expanded_id_to_hidden_state,
        expanded_id_to_children_length,
        expanded_id_to_children_node_ids,
        node_id_to_prior,
        node_id_to_parent_action,
        node_id_expanded,
        node_id_to_parent_id
):
    expanded_id = num_of_expanded_nodes
    node_id_to_expanded_id[node_id] = num_of_expanded_nodes
    num_of_expanded_nodes += 1
    node_id_to_player[expanded_id] = to_play

    num_new_nodes = valid_actions.shape[0]
    new_node_ids = np.arange(0, num_new_nodes, 1, np.uint32) + num_of_nodes

    node_id_to_reward[node_id] = reward if to_play == 0 else -reward
    expanded_id_to_hidden_state[expanded_id] = hidden_state
    expanded_id_to_children_length[expanded_id] = num_new_nodes
    expanded_id_to_children_node_ids[expanded_id, 0:num_new_nodes] = new_node_ids
    node_id_to_parent_id[new_node_ids] = node_id

    node_id_to_prior[new_node_ids] = policy_logits[valid_actions]
    node_id_to_parent_action[new_node_ids] = valid_actions
    node_id_expanded[node_id] = True
    num_of_nodes += num_new_nodes

    return num_of_expanded_nodes, num_of_nodes


@njit
def select_child_static(
        node,

        node_id_expanded,
        node_id_to_expanded_id,
        expanded_id_to_children_node_ids,
        expanded_id_to_children_length,
        node_id_to_visit_count,
        node_id_to_prior,
        max_value,
        min_value,
        pb_c_base,
        pb_c_init,
        node_id_to_parent_action,
        node_id_to_un_normalized_value,
        nodes_to_ignore,
):
    assert (node_id_expanded[node])
    expanded_id = node_id_to_expanded_id[node]
    children_ids = expanded_id_to_children_node_ids[expanded_id, 0:expanded_id_to_children_length[expanded_id]]
    nodes_to_ignore = nodes_to_ignore[children_ids]
    ucb_values = get_children_ucb(
        node,
        children_ids,

        node_id_to_visit_count,
        node_id_to_prior,
        node_id_to_un_normalized_value,

        max_value,
        min_value,

        pb_c_base,
        pb_c_init,
    )
    ucb_values[nodes_to_ignore] = FLOAT_MIN
    max_ids = np.flatnonzero(ucb_values == np.max(ucb_values))
    rand_idx = random.randint(0, max_ids.shape[0] - 1)
    selected_node = children_ids[max_ids[rand_idx]]
    return node_id_to_parent_action[selected_node], selected_node


@njit
def back_propagate_static(
        search_path: np.ndarray,
        value,

        node_id_to_value_sum,
        node_id_to_visit_count,
        node_id_to_reward,
        discount,
        min_value,
        max_value,
        node_id_to_un_normalized_value,
):
    value = value

    for i in range(search_path.shape[0] - 1, -1, -1):
        node_id = search_path[i]
        node_id_to_value_sum[node_id] += value
        node_id_to_visit_count[node_id] += 1
        value = node_id_to_reward[node_id] + discount * get_node_value(node_id, node_id_to_value_sum,
                                                                       node_id_to_visit_count)
        node_id_to_un_normalized_value[node_id] = value
        min_value = min(min_value, value)
        max_value = max(max_value, value)
    return min_value, max_value


@njit
def select_leaf_node_to_expand(
        root_node_id,
        search_path,
        initial_to_play,
        initial_depth,

        node_id_expanded,
        node_id_to_expanded_id,
        expanded_id_to_children_node_ids,
        expanded_id_to_children_length,
        node_id_to_visit_count,
        node_id_to_prior,
        max_value,
        min_value,
        pb_c_base,
        pb_c_init,
        node_id_to_parent_action,
        node_id_to_un_normalized_value,
        nodes_to_ignore,
        invert
):
    virtual_to_play = initial_to_play
    node = root_node_id
    search_path[0] = root_node_id
    search_path_length = 1
    action = -1

    while node_id_expanded[node]:
        action, node = select_child_static(
            node,

            node_id_expanded,
            node_id_to_expanded_id,
            expanded_id_to_children_node_ids,
            expanded_id_to_children_length,
            node_id_to_visit_count,
            node_id_to_prior,
            max_value,
            min_value,
            pb_c_base,
            pb_c_init,
            node_id_to_parent_action,
            node_id_to_un_normalized_value,
            nodes_to_ignore
        )
        search_path[search_path_length] = node
        search_path_length += 1

        virtual_to_play = depth_to_player_turn(search_path_length + initial_depth - 1, invert)
    return node, action, virtual_to_play, search_path_length


@njit
def select_leaf_nodes_to_expand(
        root_node_id,
        search_paths,
        current_depth,

        node_id_expanded,
        node_id_to_expanded_id,
        expanded_id_to_children_node_ids,
        expanded_id_to_children_length,
        node_id_to_visit_count,
        node_id_to_prior,
        max_value,
        min_value,
        pb_c_base,
        pb_c_init,
        node_id_to_parent_action,
        node_id_to_un_normalized_value,
        nodes_to_ignore,
        batch_size,
        invert
):
    node_ids = np.full(batch_size, root_node_id)
    action = np.full(batch_size, -1, np.uint16)
    virtual_to_play = np.full(batch_size, depth_to_player_turn(current_depth, invert))
    search_length = np.ones(batch_size, np.uint32)
    search_paths[0] = root_node_id
    for i in range(batch_size):
        node_ids[i], action[i], virtual_to_play[i], search_length[i] = select_leaf_node_to_expand(
            root_node_id,
            search_paths[i],
            virtual_to_play[i],
            current_depth,

            node_id_expanded,
            node_id_to_expanded_id,
            expanded_id_to_children_node_ids,
            expanded_id_to_children_length,
            node_id_to_visit_count,
            node_id_to_prior,
            max_value,
            min_value,
            pb_c_base,
            pb_c_init,
            node_id_to_parent_action,
            node_id_to_un_normalized_value,
            nodes_to_ignore,
            invert
        )
        nodes_to_ignore[node_ids[i]] = True
    nodes_to_ignore[node_ids] = False
    return node_ids, action, virtual_to_play, search_length


@njit
def batched_expand_and_propagate(
        nodes: np.ndarray,
        to_plays: np.ndarray,
        reward: np.ndarray,
        batched_policy_logits: np.ndarray,
        hidden_states: np.ndarray,
        search_path: np.ndarray,
        search_path_lengths,
        values: np.ndarray,
        batches,

        num_of_nodes,
        num_of_expanded_nodes,
        node_id_to_expanded_id,
        node_id_to_player,
        node_id_to_reward,
        expanded_id_to_hidden_state,
        expanded_id_to_children_length,
        expanded_id_to_children_node_ids,
        node_id_to_prior,
        node_id_to_parent_action,
        node_id_expanded,
        node_id_to_parent_id,
        node_id_to_value_sum,
        node_id_to_visit_count,
        discount,
        min_value,
        max_value,
        node_id_to_un_normalized_value,
        action_space
):
    for i in range(batches):
        num_of_expanded_nodes, num_of_nodes = expand_node_static(
            nodes[i],
            action_space,
            to_plays[i],
            reward[i],
            batched_policy_logits[i],
            hidden_states[i],

            num_of_nodes,
            num_of_expanded_nodes,
            node_id_to_expanded_id,
            node_id_to_player,
            node_id_to_reward,
            expanded_id_to_hidden_state,
            expanded_id_to_children_length,
            expanded_id_to_children_node_ids,
            node_id_to_prior,
            node_id_to_parent_action,
            node_id_expanded,
            node_id_to_parent_id
        )
        min_value, max_value = back_propagate_static(
            search_path[i, 0:search_path_lengths[i]],
            values[i],
            node_id_to_value_sum,
            node_id_to_visit_count,
            node_id_to_reward,
            discount,
            min_value,
            max_value,
            node_id_to_un_normalized_value
        )
    return num_of_expanded_nodes, num_of_nodes, min_value, max_value


class MCTS:
    def __init__(
            self,
            size_of_action_space,
            hidden_state_size,
            max_num_simulations=800,
            pb_c_base=19652,
            pb_c_init=1.25,
            discount=0.997,
    ):
        max_number_of_nodes = size_of_action_space * (max_num_simulations + 1)
        self.num_of_nodes = 1  # we will always have the root node provided by the run function
        self.num_of_expanded_nodes = 0
        self.node_id_to_visit_count = np.zeros(max_number_of_nodes,
                                               dtype=np.uint32)  # dtype should be based on the number of simulations
        self.node_id_to_player = np.full(max_num_simulations + 1, -1, dtype=np.uint8)
        self.node_id_to_value_sum = np.zeros(max_number_of_nodes, dtype=np.float32)
        self.node_id_to_prior = np.zeros(max_number_of_nodes, dtype=np.float32)
        self.node_id_expanded = np.zeros(max_number_of_nodes, dtype=bool)
        self.node_id_to_expanded_id = np.full(max_number_of_nodes, -1,
                                              dtype=np.uint16)  # uint16 because number of sim < 65000
        self.expanded_id_to_children_length = np.full(max_num_simulations + 1, -1,
                                                      dtype=np.uint16)  # uint16 because action space len < 65000
        # TODO the dtype for this array is based on the max number of actions
        self.node_id_to_parent_action = np.zeros(max_number_of_nodes, dtype=np.uint16)
        self.expanded_id_to_children_node_ids = np.zeros((max_num_simulations + 1, size_of_action_space),
                                                         dtype=np.uint32)
        # TODO currently the hidden state is a numpy float array (what to do later)
        self.expanded_id_to_hidden_state = np.zeros((max_num_simulations + 1, hidden_state_size), dtype=np.float32)
        self.action_space = np.arange(0, size_of_action_space, dtype=np.uint16)
        self.node_id_to_reward = np.zeros(max_number_of_nodes, dtype=np.float32)
        self.node_id_to_parent_id = np.zeros(max_number_of_nodes, dtype=np.uint32)
        self.node_id_to_un_normalized_value = np.zeros(max_number_of_nodes, dtype=np.float32)
        self.nodes_to_ignore = np.zeros(max_number_of_nodes, dtype=np.bool_)
        self.min_value = float("inf")
        self.max_value = -float("inf")
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.discount = discount
        self.max_num_simulations = max_num_simulations
        self.max_tree_depth = max_num_simulations

    def reinit(self):
        self.num_of_nodes = 1
        self.node_id_to_visit_count[:] = 0
        self.node_id_to_player[:] = np.iinfo(np.uint8).max
        self.node_id_to_expanded_id[:] = np.iinfo(np.int16).max  # uint16 because number of sim < 65000
        self.node_id_to_value_sum[:] = 0
        self.node_id_to_prior[:] = 0
        self.node_id_expanded[:] = 0
        self.node_id_to_parent_action[:] = 0
        self.node_id_to_reward[:] = 0
        self.min_value = float("inf")
        self.max_value = -float("inf")
        self.num_of_expanded_nodes = 0
        self.expanded_id_to_children_length[:] = 0  # uint16 because action space len < 65000
        self.expanded_id_to_children_node_ids[:] = 0
        self.expanded_id_to_hidden_state[:] = 0
        self.node_id_to_un_normalized_value[:] = 0

    def current_number_of_nodes(self):
        return self.num_of_nodes

    def expand_node(self, node_id, valid_actions: np.ndarray, to_play: int, reward: float, policy_logits: np.ndarray,
                    hidden_state: np.ndarray):
        self.num_of_expanded_nodes, self.num_of_nodes = expand_node_static(
            node_id,
            valid_actions,
            to_play, reward,
            policy_logits,
            hidden_state,

            self.num_of_nodes,
            self.num_of_expanded_nodes,
            self.node_id_to_expanded_id,
            self.node_id_to_player,
            self.node_id_to_reward,
            self.expanded_id_to_hidden_state,
            self.expanded_id_to_children_length,
            self.expanded_id_to_children_node_ids,
            self.node_id_to_prior,
            self.node_id_to_parent_action,
            self.node_id_expanded,
            self.node_id_to_parent_id
        )

    def add_noise_to_node(self, node_id, alpha=0.3, frac=0.25):
        assert (self.node_id_expanded[node_id])
        expanded_id = self.node_id_to_expanded_id[node_id]
        number_of_children = self.expanded_id_to_children_length[expanded_id]
        children_ids = self.expanded_id_to_children_node_ids[expanded_id][0:number_of_children]
        noise = np.random.dirichlet(np.full(number_of_children, alpha))
        self.node_id_to_prior[children_ids] = self.node_id_to_prior[children_ids] * (1 - frac) + noise * frac

    def select_child(self, node):
        return select_child_static(
            node,

            self.node_id_expanded,
            self.node_id_to_expanded_id,
            self.expanded_id_to_children_node_ids,
            self.expanded_id_to_children_length,
            self.node_id_to_visit_count,
            self.node_id_to_prior,
            self.max_value,
            self.min_value,
            self.pb_c_base,
            self.pb_c_init,
            self.node_id_to_parent_action,
            self.node_id_to_un_normalized_value,
            self.nodes_to_ignore
        )

    def back_propagate(self, search_path: np.ndarray, value):
        self.min_value, self.max_value = back_propagate_static(
            search_path,
            value,
            self.node_id_to_value_sum,
            self.node_id_to_visit_count,
            self.node_id_to_reward,
            self.discount,
            self.min_value,
            self.max_value,
            self.node_id_to_un_normalized_value
        )

    @timeit
    def run(self, model, initial_observation, legal_actions, current_depth, invert=False):
        to_play = depth_to_player_turn(current_depth, invert)
        root_node_id = 0
        root_value, reward, policy_logits, hidden_state = model.infer(initial_observation)
        self.expand_node(root_node_id, legal_actions, to_play, reward[0], policy_logits,
                         hidden_state)  # TODO Fix this(reward[0]) later on
        self.add_noise_to_node(root_node_id)
        # max_tree_depth = 0
        search_path = np.zeros(self.max_tree_depth, dtype=np.uint32)
        search_path[0] = root_node_id
        self.back_propagate(search_path[0:1], root_value[0])

        # when batching this make sure number of simulations is divided by the batch number
        for _ in range(self.max_num_simulations):
            node, action, virtual_to_play, search_path_length = select_leaf_node_to_expand(
                root_node_id,
                search_path,
                to_play,
                current_depth,

                self.node_id_expanded,
                self.node_id_to_expanded_id,
                self.expanded_id_to_children_node_ids,
                self.expanded_id_to_children_length,
                self.node_id_to_visit_count,
                self.node_id_to_prior,
                self.max_value,
                self.min_value,
                self.pb_c_base,
                self.pb_c_init,
                self.node_id_to_parent_action,
                self.node_id_to_un_normalized_value,
                self.nodes_to_ignore,
                invert
            )
            parent_id = search_path[search_path_length - 2]
            parent_expanded_id = self.node_id_to_expanded_id[parent_id]
            value, reward, policy_logits, hidden_state = model.infer(
                self.expanded_id_to_hidden_state[parent_expanded_id],
                action
            )
            if to_play != virtual_to_play:
                reward *= -1
            self.expand_node(node, self.action_space, virtual_to_play, reward[0], policy_logits, hidden_state)
            self.back_propagate(search_path[0:search_path_length], value[0])  # fix this value thing here

    # @timeit
    def run_batched(self, model, initial_observation, legal_actions, current_depth, batches, invert=False):
        to_play = depth_to_player_turn(current_depth, invert)
        root_node_id = 0
        root_value, reward, policy_logits, hidden_state = model.init_infer(initial_observation)
        self.expand_node(root_node_id, legal_actions, to_play, reward[0], policy_logits,
                         hidden_state)  # TODO Fix this(reward[0]) later on
        self.add_noise_to_node(root_node_id)
        # max_tree_depth = 0
        search_path = np.zeros((batches, self.max_tree_depth), dtype=np.uint32)
        search_path[0] = root_node_id
        search_path_length = 1
        self.back_propagate(search_path[0, 0:search_path_length], root_value[0])

        num_simulations = self.max_num_simulations // batches

        for _ in range(num_simulations):
            nodes, actions, virtual_to_plays, search_path_lengths = select_leaf_nodes_to_expand(
                root_node_id,
                search_path,
                current_depth,

                self.node_id_expanded,
                self.node_id_to_expanded_id,
                self.expanded_id_to_children_node_ids,
                self.expanded_id_to_children_length,
                self.node_id_to_visit_count,
                self.node_id_to_prior,
                self.max_value,
                self.min_value,
                self.pb_c_base,
                self.pb_c_init,
                self.node_id_to_parent_action,
                self.node_id_to_un_normalized_value,
                self.nodes_to_ignore,
                batches,
                invert
            )
            parent_ids = self.node_id_to_parent_id[nodes]
            parent_expanded_ids = self.node_id_to_expanded_id[parent_ids]
            values, rewards, batched_policy_logits, hidden_states = model.infer(
                self.expanded_id_to_hidden_state[parent_expanded_ids],
                actions
            )
            rewards[virtual_to_plays != to_play] *= -1
            # fix this value thing here
            self.num_of_expanded_nodes, self.num_of_nodes, self.min_value, self.max_value = batched_expand_and_propagate(
                nodes,
                virtual_to_plays,
                rewards,
                batched_policy_logits,
                hidden_states,
                search_path,
                search_path_lengths,
                values,
                batches,

                self.num_of_nodes,
                self.num_of_expanded_nodes,
                self.node_id_to_expanded_id,
                self.node_id_to_player,
                self.node_id_to_reward,
                self.expanded_id_to_hidden_state,
                self.expanded_id_to_children_length,
                self.expanded_id_to_children_node_ids,
                self.node_id_to_prior,
                self.node_id_to_parent_action,
                self.node_id_expanded,
                self.node_id_to_parent_id,
                self.node_id_to_value_sum,
                self.node_id_to_visit_count,
                self.discount,
                self.min_value,
                self.max_value,
                self.node_id_to_un_normalized_value,
                self.action_space
            )

    def select_root_action(self, temperature_thresh):
        root_node_id = 0
        expanded_id = self.node_id_to_expanded_id[root_node_id]
        num_children = self.expanded_id_to_children_length[expanded_id]
        children = self.expanded_id_to_children_node_ids[expanded_id][:num_children]
        actions = self.node_id_to_parent_action[children]
        visit_counts = self.node_id_to_visit_count[children]
        if temperature_thresh == 0:
            return actions[np.argmax(visit_counts)]
        elif temperature_thresh == float("inf"):
            return np.random.choice(actions)
            # See paper appendix Data Generation
        visit_count_distribution = visit_counts ** (1 / temperature_thresh)
        visit_count_distribution = visit_count_distribution / np.sum(visit_count_distribution)
        return np.random.choice(actions, p=visit_count_distribution)


def main():
    model = Model()
    mcts = MCTS(
        780,
        32,
        800
    )
    # test = model.infer(None)
    mcts.reinit()
    legal_actions = np.arange(0, 312, dtype=np.uint16)
    # for i in range(10):
    #     # mcts.run_batched(model, None, legal_actions, 1, 16)
    #     # mcts.reinit()
    #     mcts.run(model, None, legal_actions, 1)
    #     mcts.reinit()
    # #
    mcts.reinit()
    mcts.run_batched(model, None, legal_actions, 0, 16)
    print(mcts.select_root_action(0.0))
    # mcts.run(model, None, legal_actions, 1)
    # mcts.reinit()
    # tree = Tree()
    # tree.create_node(0, 0)
    # for i in tqdm(range(1, mcts.num_of_nodes)):
    #     tree.create_node(i, i, mcts.node_id_to_parent_id[i])
    # tree.save2file("test3")
    # tree.show()

    # pr = cProfile.Profile()
    # pr.enable()
    # for i in range(10):
    #     mcts.reinit()
    #     mcts.run(model, None, legal_actions, 0)
    # pr.disable()
    # pr.print_stats(sort='time')


if __name__ == '__main__':
    main()
