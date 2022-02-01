# rlagents.py
"""
This file defines a set of reinforcement learning agent classes.
The base class is RLAgent, all other classes inherit from that.
NOTE: some of the methods in some classes are incomplete!
Your job is to complete them.

AUTHOR: William Podlkaski (2018)
EDITORS: A Antrobus (2019)/ Annik Carson (2020)
"""
import numpy as np
from tut_basics import twoD2oneD, oneD2twoD


class RLAgent(object):
    actionlist = np.array(['D','U','R','L','J'])
    action_dict = {'D':0, 'U':1, 'R':2, 'L':3, 'J':4}

    def __init__(self, world):
        # set up environment agent is in 
        self.world = world
        # initialize state value function
        self.v = np.zeros((self.world.nstates,))
        # initialize state,action value function
        self.q = np.zeros((self.world.nstates,5))  # one column for each action
        # initialize policy (default to random)
        self.policy = self.randPolicy

    # reset agent for new runs
    def reset(self):
        self.world.init()
        self.state = self.world.get_state()

    # choose actions
    def choose_action(self):
        state = self.world.get_state()
        actions = self.world.get_actions()
        self.action = self.policy(state, actions)
        return self.action

    # perform actions, get feedback from environment
    def take_action(self, action):
        (self.state, self.reward, terminal) = self.world.move(action)
        return terminal

    ## -------------------
    ## Available Policies
    ## -------------------
    def randPolicy(self, state, actions):
        # np.random.choice() is a uniform distribution over actions
        available_actions = self.actionlist[actions]
        return np.random.choice(available_actions)

    def greedyQPolicy(self, state, actions):
        idx = np.arange(5)[actions]
        return self.actionlist[idx[np.argmax(self.q[state,actions])]]

    def epsilongreedyQPolicy(self, state, actions, epsilon=0.2):
        idx = np.arange(5)[actions]
        greedy_action = self.actionlist[idx[np.argmax(self.q[state,actions])]]
        nongreedy_actions = np.delete(self.actionlist[actions],np.argwhere(self.actionlist==greedy_action))
        r = np.random.rand()
        for c in range(len(nongreedy_actions)):
            if (r<((c+1)*epsilon/len(nongreedy_actions))):
                return nongreedy_actions[c]
        return greedy_action


class RLExampleAgent(RLAgent):
    def __init__(self, world):
        super(RLExampleAgent, self).__init__(world)
        self.v = np.random.normal(size=world.nstates)
        self.q = np.random.normal(size=(world.nstates,5))
        self.Ppi = np.zeros((world.nstates,world.nstates))
        rows = np.array([1,2,3,4, 5,6,8, 11,13,15,16,18])
        cols = np.array([2,3,8,15,6,1,13,6, 18,16,11,19])
        self.Ppi[rows,cols] = 1




