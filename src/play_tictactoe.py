import numpy as np

from src.utils import reshape_state


def play_tictactoe(env, qtable, state_dict, max_steps=9, num_test_games=3):
    """
    play against the trained Q-Learning agent
    Args:
        env (class): environment class
        qtable (array): numpy array containing the qtable respect. the state-action values
        state_dict (dict): encoding of the state array
        max_steps (int): max steps to take in one episode
        num_test_games (int): number of times to play against the trained agent
    """

    player1 = 1
    player2 = 2

    for _ in range(num_test_games):
        

        action_space = np.arange(9)
        start = np.random.choice([1,2])  # 0 or 1

        state, _ = env.reset()
        state = np.append(state, start)

        state = state_dict[reshape_state(state)]

        if start == 0:
            print("Human beginns")
        else:
            print("Agent beginns")
        print("--" * 10)
        print(env.render())

        for _step in range(start, max_steps + start):
            # alternate the moves of the players
            if _step % 2 == 0:
                
                print("--" * 10)
                print("Move Human")
                action = np.nan

                while action not in action_space:
                    action = int(input(f"choose an action from {action_space}:"))
                    print("Action:", action)
                action_space = action_space[action_space != action]
                state, reward, done, _ = env.step((action, player1))
                state = np.append(state, player1)
                state = state_dict[reshape_state(state)]
                print(env.render())
                print(reward)
                if done:
                    print("**" * 10)
                    print("Human won!")
                    print("**" * 10)
                    print(env.render())
                    print("\n" * 2)
                    break
            else:
                print("--" * 10)
                print("move Agent")

                action = max(action_space, key=lambda action: qtable[state, action])
                # array = np.array(qtable[state, :])
                # order = array.argsort()
                # ranks = order.argsort()
                # max_value_rank = np.min(ranks[action_space])
                # action = np.where(ranks == max_value_rank)[0][0]

                print("Action:", action)
                action_space = action_space[action_space != action]
                state, reward, done, _ = env.step((action, player2))
                state = np.append(state, player2)
                state = state_dict[reshape_state(state)]
                print(env.render())
                if done:
                    print("**" * 10)
                    print("Agent won!")
                    print("**" * 10)
                    print(env.render())
                    print("\n" * 2)
                    break

            print("\n")
            # stopping criterion
            if not done and _step == max_steps + start - 1:
                if reward != env.large - 1:
                    print("There is no Winner!")
                    print("--" * 10)
                    print("--" * 10)
                    print("\n" * 2)
                break


def play_tictactoe_with_random(env, qtable, state_dict, max_steps=9, num_test_games=3, quiet=False):
    """
    play against the trained Q-Learning agent
    Args:
        env (class): environment class
        qtable (array): numpy array containing the qtable respect. the state-action values
        state_dict (dict): encoding of the state array
        max_steps (int): max steps to take in one episode
        num_test_games (int): number of times to play against the trained agent
    """

    player1 = 1
    player2 = 2
    agent_win_history = []
    total_reward = 0

    for _ in range(num_test_games):
        

        action_space = np.arange(9)

        start = np.random.choice([1,2])

        state, _ = env.reset()
        state = np.append(state, start)
        state = state_dict[reshape_state(state)]

        # if start == 1:
        #     print("Random beginns")
        # else:
        #     print("Agent beginns")
        # print("--" * 10)

        for _step in range(start, max_steps + start):

            # alternate the moves of the players
            if _step % 2 == 0:
                # print(env.render())
                # print("--" * 10)
                # print("Move RandomBot")
                action = np.random.choice(action_space)
                action_space = action_space[action_space != action]

                state, reward, done, _ = env.step((action, player1))
                state = np.append(state, player1)
                state = state_dict[reshape_state(state)]
                # print("Action:", action)
                if done:
                    # print("**" * 10)
                    # print("Random won!")
                    # print("**" * 10)
                    # print(env.render())
                    # print("\n" * 2)
                    agent_win_history.append(-1)
                    break
            else:
                # print(env.render())
                # print("--" * 10)
                # print("move Agent")

                # array = np.array(qtable[state, :])
                # order = array.argsort()
                # ranks = order.argsort()
                # max_value_rank = np.min(ranks[action_space])
                # action = np.where(ranks == max_value_rank)[0][0]
                action = max(action_space, key=lambda action: qtable[state, action])

                # print("Action:", action)
                action_space = action_space[action_space != action]

                state, reward, done, _ = env.step((action, player2))
                total_reward += reward
                state = np.append(state, player2)
                state = state_dict[reshape_state(state)]
                # print(env.render())
                if done:
                    # print("**" * 10)
                    # print("Agent won!")
                    # print("**" * 10)
                    # print(env.render())
                    # print("\n" * 2)
                    agent_win_history.append(1)
                    break

            # print("\n")
            # stopping criterion
            if not done and _step == max_steps + start - 1:
                if reward != env.large - 1:
                    # print("There is no Winner!")
                    # print("--" * 10)
                    # print("--" * 10)
                    # print("\n" * 2)
                    agent_win_history.append(1)
                break

    return agent_win_history, total_reward
