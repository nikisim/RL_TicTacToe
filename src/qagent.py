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

    def update_qtable(self, state, new_state, action, reward, done):
        """
        Q-Value update:
        Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """
        reference_qvalue = reward + self.gamma * np.max(self.qtable[new_state, :])
        updated_qvalue = self.learning_rate * reference_qvalue + (1 - self.learning_rate) * self.qtable[state, action]
        return updated_qvalue
    
    def get_qtable(self):
        return self.qtable

    def update_epsilon(self, episode: int):
        self.epsilon = (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode
                                ) + self.min_epsilon

    def get_action(self, state: int, action_space: np.array) -> int:
        """
        Compute the action to take in the current state, including exploration.  
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action.
        """
        if random.random() < self.epsilon:
            action = np.random.choice(action_space)
        else:
            action = max(action_space, key=lambda action: self.qtable[state, action])
            
        return action
