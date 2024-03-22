from src.play_tictactoe import play_tictactoe
from src.utils import load_qtable, create_state_dictionary
import gym
import numpy as np
import sys
import warnings
import gym_TicTacToe

# ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

q_table = load_qtable('tables', "q_table_best_09")

state_dict = create_state_dictionary()

env = gym.envs.make("TTT-v0", small=-1, large=10)

if __name__ == "__main__":
    play_tictactoe(env, q_table, state_dict, num_test_games=1)
