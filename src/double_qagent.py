import random

import numpy as np


class DoubleQLearningAgent(object):
    """
    Implementation of a Double Q-learning Algorithm
    """

    def __init__(
        self,
        exploration_parameters: dict,
        state_size: int,
        action_size: int,
        learning_rate, gamma
    ):
        #random positive init of q-Table
        self.qtable1 = np.ones((state_size, action_size))*np.random.random()*0.001
        self.qtable2 = np.ones((state_size, action_size))*np.random.random()*0.001


        self.learning_rate = learning_rate
        self.gamma = gamma

        self.epsilon = exploration_parameters["max_epsilon"]
        self.max_epsilon = exploration_parameters["max_epsilon"]
        self.min_epsilon = exploration_parameters["min_epsilon"]
        self.decay_rate = exploration_parameters["decay_rate"]

    def update_qtables(self, state, new_state, action, reward, done):
        """
        Double Q-Value update:
        Randomly update one of the Q-tables using the other for the max Q-value estimate
        Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """
        if np.random.rand() < 0.5:
            best_action = np.argmax(self.qtable1[new_state, :])
            reference_qvalue = reward + self.gamma * self.qtable2[new_state, best_action]
            self.qtable1[state, action] = (1 - self.learning_rate) * self.qtable1[state, action] + self.learning_rate * reference_qvalue
        else:
            best_action = np.argmax(self.qtable2[new_state, :])
            reference_qvalue = reward + self.gamma * self.qtable1[new_state, best_action]
            self.qtable2[state, action] = (1 - self.learning_rate) * self.qtable2[state, action] + self.learning_rate * reference_qvalue
    
    def get_qtable(self):
        return self.qtable

    def update_epsilon(self, episode: int):
        self.epsilon = (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode
                                ) + self.min_epsilon

    def get_action(self, state: int, action_space: np.array) -> int:
        """
        Compute the action to take in the current state, including exploration.  
        Use the average of both Q-tables for selecting the best action.
        """
        average_qtable = (self.qtable1 + self.qtable2) / 2
        if np.random.rand() < self.epsilon:
            action = np.random.choice(action_space)
        else:
            action = np.argmax(average_qtable[state, :])
        return action
