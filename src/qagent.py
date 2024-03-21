import random

import numpy as np


class QLearningAgent(object):
    """
    Implementation of a Q-learning Algorithm
    """

    def __init__(
        self,
        exploration_parameters: dict,
        state_size: int,
        action_size: int,
        learning_rate, gamma
    ):
        #random positive init of q-Table
        self.qtable = np.ones((state_size, action_size))*np.random.random()*0.001

        self.learning_rate = learning_rate
        self.gamma = gamma

        self.epsilon = exploration_parameters["max_epsilon"]
        self.max_epsilon = exploration_parameters["max_epsilon"]
        self.min_epsilon = exploration_parameters["min_epsilon"]
        self.decay_rate = exploration_parameters["decay_rate"]

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self.qtable[state,action]
    
    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self.qtable[state,action] = value

    def get_value(self, state: int, state_dict):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.get_legal_actions(state)
        key = list(filter(lambda x: state_dict[x] == state, state_dict))[0]
        state_pure = np.array(key[:-1])
        action_space = np.where(state_pure == 0)[0]

        # If there are no legal actions, return 0.0
        if len(action_space) == 0:
            return 0.0

        return max([self.get_qvalue(state, act) for act in action_space])

    def update_qtable(self, state, new_state, action, reward, done):
        """
        Q-Value update:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """
        # return self.qtable[state, action] + self.learning_rate * (
        #     reward
        #     + self.gamma * np.max(self.qtable[new_state, :]) * (1 - done)
        #     - self.qtable[state, action])

        reference_qvalue = reward + self.gamma * np.max(self.qtable[new_state, :])
        updated_qvalue = self.learning_rate * reference_qvalue + (1 - self.learning_rate) * self.qtable[state, action]
        return updated_qvalue
    
    def get_qtable(self):
        return self.qtable

    def update_epsilon(self, episode: int):
        self.epsilon = (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode
                                ) + self.min_epsilon

    def get_action(self, state: int, action_space: np.array) -> int:
        if random.random() < self.epsilon:
            action = np.random.choice(action_space)
        else:
            action = max(action_space, key=lambda action: self.qtable[state, action])
            
        return action
