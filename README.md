### RL_TicTacToe

This repository contains a Q-Learning RL implementation for TicTacToe env from (https://github.com/MauroLuzzatto/OpenAI-Gym-TicTacToe-Environment).


Обучение агента происходит с помощью табличного Q-обучения. Во время обучения агент играет сам с собой, очередность хода меняется случайным образом.


Для повышения эффективности к ветору состояния добавлена метка хода (1 или 2) для того, чтобы агент понимал, чей ход в конкретной ситуации.

## How to install
### 1) Setup
```
git clone git@github.com:MauroLuzzatto/OpenAI-Gym-TicTacToe-Environment.git
cd OpenAI-Gym-TicTacToe-Environment
pip install -r requirements.txt
```

### 2) Register the Environment

run the following command in the command line
``` 
pip install -e gym-TicTacToe/. 
```


## How to play against the agent

```
python3 play_to_test
```