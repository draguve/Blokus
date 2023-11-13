import math

import numpy as np
from .games.blokus import Game
from numba import njit


# @njit(parallel=True)
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


def get_children_value(children_node_ids, id_to_value_sum, id_to_visit_count):
    values = np.zeros(children_node_ids.shape[0])
    indexes_where_positive = (id_to_visit_count[children_node_ids] > 0).nonzero()
    ids_where_positive = children_node_ids[indexes_where_positive]
    values[indexes_where_positive] = id_to_value_sum[ids_where_positive] / id_to_visit_count[ids_where_positive]
    return values


def get_node_value(node_id, id_to_value_sum, id_to_visit_count):
    if id_to_visit_count[node_id] == 0:
        return 0
    return id_to_value_sum[node_id] / id_to_visit_count[node_id]


class MCTS:
    def __init__(self, max_number_of_nodes, size_of_action_space, hidden_state_size, pb_c_base, pb_c_init,
                 discount, max_num_simulations):
        self.node_index = 0
        self.node_id_to_visit_count = np.zeros(max_number_of_nodes, dtype=int)
        self.node_id_to_player = np.full(max_number_of_nodes, -1, dtype=np.uint8)
        self.node_id_to_value_sum = np.zeros(max_number_of_nodes, dtype=float)
        self.node_id_to_prior = np.zeros(max_number_of_nodes, dtype=float)
        self.node_id_expanded = np.zeros(max_number_of_nodes, dtype=bool)
        self.node_id_to_children_length = np.full(max_number_of_nodes, -1, dtype=np.uint16)
        # TODO the dtype for this array is based on the max number of nodes
        self.node_id_to_parent_action = np.zeros(max_number_of_nodes, dtype=np.uint16)
        self.node_id_to_children_node_ids = np.zeros((max_number_of_nodes, size_of_action_space), dtype=np.uint16)
        # TODO currently the hidden state is a numpy float array (what to do later)
        self.node_id_to_hidden_state = np.zeros((max_number_of_nodes, hidden_state_size), dtype=np.float32)
        self.action_space = np.arange(0, size_of_action_space, dtype=np.uint16)
        self.node_id_to_reward = np.zeros(max_number_of_nodes, dtype=float)
        self.min_value = float("inf")
        self.max_value = -float("inf")
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.discount = discount
        self.max_num_simulations = max_num_simulations
        self.max_tree_depth = max_num_simulations

    def current_number_of_nodes(self):
        return self.node_index + 1

    def expand_node(self, node_id, valid_actions: np.ndarray, to_play: int, reward: float, policy_logits: np.ndarray,
                    hidden_state: np.ndarray):
        self.node_id_to_player[node_id] = to_play
        self.node_id_to_reward[node_id] = reward
        self.node_id_to_hidden_state[node_id] = hidden_state

        num_new_nodes = valid_actions.shape[0]
        self.node_id_to_children_length[node_id] = num_new_nodes
        new_node_ids = np.arange(0, num_new_nodes, dtype=int) + self.node_index
        self.node_id_to_children_node_ids[node_id, 0:num_new_nodes] = new_node_ids
        self.node_id_to_prior[new_node_ids] = policy_logits[valid_actions]
        self.node_id_to_parent_action[new_node_ids] = valid_actions
        self.node_index += num_new_nodes

    def add_noise_to_node(self, node_id, alpha, frac):
        children_ids = self.node_id_to_children_node_ids[node_id]
        noise = np.random.dirichlet(np.full(self.node_id_to_children_length[node_id], alpha))
        self.node_id_to_prior[children_ids] = self.node_id_to_prior[children_ids] * (1 - frac) + noise * frac

    def select_child(self, node):
        children_ids = self.node_id_to_children_node_ids[0:self.node_id_to_children_length[node]]
        ucb_values = get_children_ucb(
            node, children_ids, self.node_id_to_visit_count, self.node_id_to_prior,
            self.node_id_to_value_sum, self.node_id_to_reward, self.max_value, self.min_value,
            self.pb_c_base, self.pb_c_init, self.discount
        )
        max_ids = children_ids[np.argmax(ucb_values)]
        selected_node = np.random.choice(max_ids)
        return self.node_id_to_parent_action[selected_node], selected_node

    def back_propagate(self, search_path: np.ndarray, value):
        value = value
        for i in reversed(range(search_path.shape[0])):
            node_id = search_path[i]
            self.node_id_to_value_sum[node_id] += value
            self.node_id_to_visit_count[node_id] += 1
            value = self.node_id_to_reward[node_id] + self.discount * get_node_value(node_id, self.node_id_to_value_sum,
                                                                                     self.node_id_to_visit_count)
            self.min_value = min(self.min_value, value)
            self.max_value = max(self.max_value, value)

    def run(self, model, initial_observation, legal_actions, to_play):
        root_node_id = 0
        root_value, reward, policy_logits, hidden_state = model.infer(initial_observation)
        self.expand_node(root_node_id, legal_actions, to_play, reward, policy_logits, hidden_state)

        max_tree_depth = 0
        search_path = np.zeros(self.max_tree_depth, dtype=np.uint16)
        search_path_max = 0
        for _ in range(self.max_num_simulations):
            virtual_to_play = to_play
            node = root_node_id
            search_path[0] = root_node_id
            search_path_max = 1
            action=-1

            while self.node_id_expanded[node]:
                action, node = self.select_child(node)
                search_path[search_path_max] = node
                search_path_max += 1

                virtual_to_play = #how to fix this, dont know yet

            parent_id = search_path[search_path_max-2]
            value, reward, policy_logits, hidden_state = model.recurrent_inference(
                self.node_id_to_hidden_state[parent_id],
                action
            )
            if to_play != virtual_to_play:
                reward *= -1
            self.expand_node(node,self.action_space,virtual_to_play,reward,policy_logits,hidden_state)
            self.back_propagate(search_path[0:search_path_max],value)
