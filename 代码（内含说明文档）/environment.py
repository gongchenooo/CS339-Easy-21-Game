import random
import numpy as np

class State:
    dealercard = random.randint(1, 10) # dealer的初始值
    playersum = random.randint(1, 10) # player的sum

def draw_card(current):
    card_num = random.randint(1, 10)
    if random.randint(1,3) < 3:
        current += card_num # 黑色
    else:
        current -= card_num # 红色
    return current

# action {0: stick, 1: hit}
# return state, reward
def step(state, action):
    if action == 1:
        state.playersum = draw_card(state.playersum) # 状态改变
        if state.playersum > 21 or state.playersum < 1:
            return "terminal", -1.0
        else:
            return state, 0
    elif action == 0: #到dealer的回合
        while(state.dealercard < 16):
            state.dealercard = draw_card(state.dealercard)
            if state.dealercard > 21 or state.dealercard < 1:
                return "terminal", 1.0
        if state.dealercard > state.playersum:
            return "terminal", -1.0
        elif state.dealercard < state.playersum:
            return "terminal", 1.0
        else:
            return "terminal", 0.0


