# Starter code for assignment 8
# CS 152: Neural Networks
# HMC, Fall, 2018

import argparse
import collections
import copy
import numpy as np
import random
import statistics

class Gridworld():
    ACTIONS = ['^', '>', 'v', '<']

    def __init__(self, rows, cols, start_state, terminal_states):
        self.rows = rows
        self.cols = cols
        self.start_state = start_state
        self.terminals = terminal_states
        self.reset()

    def reset(self):
        self.current_state = self.start_state
        self.total_reward = 0

    def state(self):
        return self.current_state

    def set_state(self, state):
        self.current_state = state

    def done(self):
        return self.current_state in self.terminals

    def states(self):
        return [(col, row) for col in range(self.cols) for row in range(self.rows)]

    def actions(self):
        return self.ACTIONS
                
    def terminal_states(self):
        return self.terminals

    def print_states(self, state_values):
        for row in range(self.rows):
            for col in range(self.cols):
                state = (col, row)
                print(state_values[state], end=' ')
            print()

class WindyGridworld(Gridworld):
    WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

    def __init__(self, rows=7, cols=10):
        start_state = (0, rows // 2)
        terminal_states = [(cols - 3, rows // 2)]
        super().__init__(rows=rows, cols=cols, start_state=start_state, terminal_states=terminal_states)

    def step(self, a):
        """Returns (state,reward) pair."""
        col, row = self.current_state
        new_col, new_row = col, row
        if a == '^':
            new_row = row - 1
        elif a == 'v':
            new_row = row + 1
        elif a == '>':
            new_col = col + 1
        elif a == '<':
            new_col = col -1
        else:
            raise ValueError(f"invalid action: {a}")
        new_col = max(0, min(new_col, self.cols-1))
        new_row -= self.WIND[col]
        new_row = max(0, min(new_row, self.rows-1))
        self.current_state = (new_col, new_row)
        return (self.current_state, -1)


class Cliff(Gridworld):

    def __init__(self, rows=4, cols=12):
        start_state=(0, rows-1)
        terminal_states = [(cols-1, rows-1)]
        super().__init__(rows=rows, cols=cols, start_state=start_state, terminal_states=terminal_states)

    def step(self, a):
        col, row = self.current_state
        if a == '^':
            row = row - 1
        elif a == 'v':
            row = row + 1
        elif a == '>':
            col = col + 1
        elif a == '<':
            col = col -1
        else:
            raise ValueError(f"invalid action: {a}")
        col = max(0, min(col, self.cols-1))
        row = max(0, min(row, self.rows-1))
        if row == self.rows-1 and col > 0 and col < self.cols-1:
            self.current_state = self.start_state
            reward = -100
        else:
            self.current_state = (col, row)
            reward = -1
        return (self.current_state, reward)

class EpsilonGreedyActor():
    def __init__(self, actor, actions, epsilon=0.1):
        self.actor = actor
        self.actions = actions
        self.epsilon = epsilon

    def action(self, state):
        r = random.random()
        if r < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.actor.action(state)

class GreedyActor():
    def __init__(self, q):
        self.q = q

    def action(self, state):
        return key_with_max_value(self.q[state])


def key_with_max_value(d):
    return max(d, key=d.get)


def create_q(env):
    q = {}
    for state in env.states():
        q[state] = {action:0 for action in env.actions()}
    return q


def print_q(env, q, cycle):
    print(f"Q_𝜋_{cycle} in order {env.actions()}:")
    to_print = {}
    for state, actions in q.items():
        to_print[state] = '(' + " ".join([f'{actions[action]:5.2f}' for action in env.actions()]) + ')'
    env.print_states(to_print)

def print_pi(env, actor, cycle):
    print(f"𝜋_{cycle}:")
    to_print = {}
    for state in env.states():
        if state in env.terminal_states():
            to_print[state] = "T"
        else:
            to_print[state] = actor.action(state)
    env.print_states(to_print)


def main(args):
    random.seed(829)
    if args.env == 'smallcliff':
        env = Cliff(cols=3)
    elif args.env == 'cliff':
        env = Cliff()
    elif args.env == 'smallwindy':
        env = WindyGridworld(cols=6)
    elif args.env == 'windy':
        env = WindyGridworld()
    else:
        raise ValueError(f'Unknown env: {args.env}')
    q = create_q(env)
    greedy_actor = GreedyActor(q)
    epsilon_greedy_actor = EpsilonGreedyActor(greedy_actor, env.actions())

    total_reward = 0
    total_reward_cycles = 0
    for cycle in range(args.num_cycles):
        env.reset()
        reward = None
        steps = 0
        while not env.done():
            state = env.state()
            action = epsilon_greedy_actor.action(state)
            if args.verbose:
                print(f"s={env.state()},a={action}, r={reward}")
            _, reward = env.step(action)
            steps += 1
            total_reward += reward
        if (cycle %10) == 0:
            print_q(env, q, cycle)
            print_pi(env, greedy_actor, cycle+1)
            print("")
        total_reward_cycles += 1
        if (cycle % 10) == 0:
            print("average reward/episode", total_reward/total_reward_cycles)
            total_reward = 0
            total_reward_cycles = 0
            print("")


parser = argparse.ArgumentParser(description='Solve a gridworld.')
parser.add_argument('--env', help='specify the environment, one of (cliff, windy, smallcliff, smallwindy)', default='cliff')
parser.add_argument('--alg', help='specify the algorithm, one of (sarsa, qlearning)', default='sarsa')
parser.add_argument('--num_cycles', help='number of cycles to run', default=5, type=int)
parser.add_argument('--verbose', help='increase output verbosity', action='store_true')
args = parser.parse_args()
main(args)

