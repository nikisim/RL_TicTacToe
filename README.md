# RL_TicTacToe

This repository contains a Q-Learning RL implementation for TicTacToe env from (https://github.com/MauroLuzzatto/OpenAI-Gym-TicTacToe-Environment).


Обучение агента происходит с помощью табличного Q-обучения с epsilon-greedy стратегией. Во время обучения агент играет сам с собой, очередность хода меняется случайным образом.

### Current Win Rate
```
Win Rate = 0.9
```

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

## About environment

### Действия
Пространство действий состоит из чисел от 0 до 8, каждое из которых определяет позицию на поле. Таблицу соответствия позиции действиям можно видеть ниже:


|  |  |  |
| :---: |  :---:  |  :---: |
| 0  | 1  | 2  |
| 3  | 4  | 5  |
| 6  | 7  | 8  |


### Состояния
Пространоство состояний состоит из 17906 состояний
State space:
-    Для доски 3х3 всего состояний может быть: 3^3^2 = 19 683 комбинации (в каждой ячейке три состояния: 0,1,2)
-    Однако, не все состояния могут быть реализованы (например, нельзя чтобы все поле было в крестиках)
-    Таким образом, пространство состояний может быть уменьшено до 8 953 состояний
-    Но каждое состояние встречается 2 раза, поэтому 8953*2=17906
-    Для повышения эффективности к ветору состояния добавлена метка хода (1 или 2) для того, чтобы агент понимал, чей ход в конкретной ситуации (без этой модификации эффективность агента уменьшается).

Описание состояний:
   
    - 0 - свободная клетка,
    - 1 - крестик
    - 2 - нолик

### Награда
За победу агенту дается вознаграждение +10, за каждый сыгранный ход штраф -1. ТАкже агент получает дополнительную награду, если блокирует победный ход противника и получает +9.

```
REWARD_LARGE = 10
REWARD_SMALL = -1
REWARD_FOR_BLOCK = 9
```


### Done
Игра заканчивается, когда:
-    Один из игроков разместил свои маркеры (крестики или нолики) либо вертикали, либо по горизонтали, либо по диагонали
-    Доска заполнена маркерами, но ни один из игроков не выиграл



## How to play against the agent
Чтобы поиграть против предобученного агента, используйте следующую команду:

```
python3 play_to_test
```

## How to check current win rate

```
python3 play_against_random
```