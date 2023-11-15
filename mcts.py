import math

import numpy as np

# from .games.blokus import Game
from numba import njit
from tqdm import tqdm
from treelib import Tree

from util import timeit


class Model:
    def __init__(self):
        pass

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def infer(self, obs, o=None):
        value = np.random.rand(1)
        reward = np.random.rand(1)
        policy_logits = np.random.rand(780)
        hidden_state = np.random.rand(32)
        return value, reward, self.softmax(policy_logits), hidden_state


# TODO Can we cache this
@njit(parallel=True)
def get_children_ucb(
        parent_node_id,
        children_node_ids,

        id_to_visit_count,
        id_to_prior,
        id_to_value_sum,
        id_to_reward,

        max_value,
        min_value,

        pb_c_base,
        pb_c_init,
        discount
):
    pb_c = math.log(
        (id_to_visit_count[parent_node_id] + pb_c_base + 1) / pb_c_base
    ) + pb_c_init

    pb_c = pb_c * np.sqrt(id_to_visit_count[parent_node_id]) / ((id_to_visit_count[children_node_ids]) + 1)
    prior_score = pb_c * id_to_prior[children_node_ids]

    rewards = id_to_reward[children_node_ids]
    children_values = get_children_value(children_node_ids, id_to_value_sum, id_to_visit_count)
    value_score = rewards + discount * children_values  # TODO ??? what to do here - or + ???
    value_score = value_score - min_value / max_value - min_value

    return prior_score + value_score


@njit(parallel=True)
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
def depth_to_player_turn(depth):
    return (depth // 2) % 2


@njit
def expand_node_static(
        node_id,
        valid_actions: np.ndarray,
        to_play: int, reward: float,
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
    node_id_to_player[node_id] = to_play

    num_new_nodes = valid_actions.shape[0]
    new_node_ids = np.arange(0, num_new_nodes, 1, np.uint32) + num_of_nodes

    node_id_to_reward[node_id] = reward
    expanded_id_to_hidden_state[expanded_id] = hidden_state
    expanded_id_to_children_length[expanded_id] = num_new_nodes
    expanded_id_to_children_node_ids[expanded_id, 0:num_new_nodes] = new_node_ids
    node_id_to_parent_id[new_node_ids] = node_id

    node_id_to_prior[new_node_ids] = policy_logits[valid_actions]
    node_id_to_parent_action[new_node_ids] = valid_actions
    node_id_expanded[node_id] = True
    num_of_nodes += num_new_nodes

    return num_of_expanded_nodes, num_of_nodes


def select_child_static(
        node,

        node_id_expanded,
        node_id_to_expanded_id,
        expanded_id_to_children_node_ids,
        expanded_id_to_children_length,
        node_id_to_visit_count,
        node_id_to_prior,
        node_id_to_value_sum,
        node_id_to_reward,
        max_value,
        min_value,
        pb_c_base,
        pb_c_init,
        discount,
        node_id_to_parent_action
):
    assert (node_id_expanded[node])
    expanded_id = node_id_to_expanded_id[node]
    children_ids = expanded_id_to_children_node_ids[expanded_id,
                   0:expanded_id_to_children_length[expanded_id]]
    ucb_values = get_children_ucb(
        node, children_ids, node_id_to_visit_count, node_id_to_prior,
        node_id_to_value_sum, node_id_to_reward, max_value, min_value,
        pb_c_base, pb_c_init, discount
    )
    max_ids = np.flatnonzero(ucb_values == np.max(ucb_values))
    selected_node = children_ids[np.random.choice(max_ids)]
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
        max_value
):
    value = value

    for i in range(search_path.shape[0], -1, -1):
        node_id = search_path[i]
        node_id_to_value_sum[node_id] += value
        node_id_to_visit_count[node_id] += 1
        value = node_id_to_reward[node_id] + discount * get_node_value(node_id, node_id_to_value_sum,
                                                                       node_id_to_visit_count)
        min_value = min(min_value, value)
        max_value = max(max_value, value)
    return min_value, max_value

def select_leaf_node_to_expand(

):



def get_nodes_to_expand(
        current_depth,
        batch_size,
        root_node_id,
        max_depth,
        node_id_expanded,
):
    virtual_to_play = np.full(batch_size, depth_to_player_turn(current_depth))
    node_ids = np.full(batch_size, root_node_id)
    search_paths = np.zeros((batch_size, max_depth), dtype=np.uint16)
    search_length = np.ones(batch_size, dtype=int)
    actions = np.full(batch_size, -1, dtype=np.uint16)
    for i in range(batch_size):
        while node_id_expanded[node_ids[i]]:
            action, node = self.select_child(node_ids[i])  # need to add a function to ignore already selected nodes
            search_paths[i][search_length[i]] = node
            search_length[i] += 1

            virtual_to_play = depth_to_player_turn(search_length[i] + current_depth - 1)


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
        max_number_of_nodes = (max_num_simulations + 1) * (max_num_simulations + 1)
        self.num_of_nodes = 1  # we will always have the root node provided by the run function
        self.num_of_expanded_nodes = 0
        self.node_id_to_visit_count = np.zeros(max_number_of_nodes,
                                               dtype=np.uint32)  # dtype should be based on the number of simulations
        self.node_id_to_player = np.full(max_number_of_nodes, -1, dtype=np.uint8)
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
        self.node_id_to_value_sum[:] = 0
        self.node_id_to_prior[:] = 0
        self.node_id_expanded[:] = 0
        self.node_id_to_parent_action[:] = 0
        self.node_id_to_reward[:] = 0
        self.min_value = float("inf")
        self.max_value = -float("inf")
        self.num_of_expanded_nodes = 0
        self.node_id_to_expanded_id[:] = np.iinfo(np.int16).max  # uint16 because number of sim < 65000
        self.expanded_id_to_children_length[:] = 0  # uint16 because action space len < 65000
        self.expanded_id_to_children_node_ids[:] = 0
        self.expanded_id_to_hidden_state[:] = 0

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

    # TODO Untested as of now
    def add_noise_to_node(self, node_id, alpha=0.3, frac=0.25):
        if self.node_id_expanded[node_id]:
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
            self.node_id_to_value_sum,
            self.node_id_to_reward,
            self.max_value,
            self.min_value,
            self.pb_c_base,
            self.pb_c_init,
            self.discount,
            self.node_id_to_parent_action
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
            self.max_value
        )

    def run(self, model, initial_observation, legal_actions, current_depth):
        to_play = depth_to_player_turn(current_depth)
        root_node_id = 0
        root_value, reward, policy_logits, hidden_state = model.infer(initial_observation)
        self.expand_node(root_node_id, legal_actions, to_play, reward[0], policy_logits,
                         hidden_state)  # TODO Fix this(reward[0]) later on
        self.add_noise_to_node(root_node_id)
        # max_tree_depth = 0
        search_path = np.zeros(self.max_tree_depth, dtype=np.uint32)

        search_path[0] = root_node_id
        search_path_length = 1

        self.back_propagate(search_path[0:search_path_length], root_value[0])

        # when batching this make sure number of simulations is divided by the batch number
        for _ in range(self.max_num_simulations):
            virtual_to_play = to_play
            node = root_node_id
            search_path[0] = root_node_id
            search_path_length = 1
            action = -1

            while self.node_id_expanded[node]:
                action, node = self.select_child(node)
                search_path[search_path_length] = node
                search_path_length += 1

                virtual_to_play = depth_to_player_turn(search_path_length + current_depth - 1)

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


def main():
    model = Model()
    mcts = MCTS(
        780,
        32,
    )
    test = model.infer(None)
    legal_actions = np.arange(0, 780, dtype=np.uint16)
    mcts.run(model, None, legal_actions, 0)
    tree = Tree()
    tree.create_node(0, 0)
    for i in tqdm(range(1, mcts.num_of_nodes)):
        tree.create_node(i, i, mcts.node_id_to_parent_id[i])
    tree.save2file("test2")
    # tree.show()
    # mcts.reinit()
    # mcts.run(model, None, legal_actions, 0)
    print("test")


if __name__ == '__main__':
    main()
