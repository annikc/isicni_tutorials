# rlagents.py
"""
This file defines a set of reinforcement learning agent classes.
The base class is RLAgent, all other classes inherit from that.
NOTE: some of the methods in some classes are incomplete!
Your job is to complete them.

AUTHOR: William Podlkaski (2018)
EDITOR: A Antrobus (2019)
"""
import numpy as np
from gridworld import twoD2oneD, oneD2twoD
from plotFunctions import plotStateActionValue

class RLAgent(object):

	actionlist = np.array(['D','U','R','L','J'])
	action_dict = {'D':0, 'U':1, 'R':2, 'L':3, 'J':4}

	def __init__(self, world):
		self.world = world
		self.v = np.zeros((self.world.nstates,))
		self.q = np.zeros((self.world.nstates,5))  # one column for each action
		self.policy = self.randPolicy

	def reset(self):
		self.world.init()
		self.state = self.world.get_state()

	def choose_action(self):
		state = self.world.get_state()
		actions = self.world.get_actions()
		self.action = self.policy(state, actions)
		return self.action

	def take_action(self, action):
		(self.state, self.reward, terminal) = self.world.move(action)
		return terminal

	def run_episode(self):
		print("Running episode...")
		is_terminal = False
		self.reset()
		c = 0
		while (is_terminal == False):
			c += 1
			prev_state = oneD2twoD(self.state,self.world.shape)
			action = self.choose_action()
			is_terminal = self.take_action(action)
			state = oneD2twoD(self.state,self.world.shape)
			print("Step %d: move from (%d,%d) to (%d,%d), reward = %.2f" % (c,prev_state[0],prev_state[1],state[0],state[1],self.reward))
		print("Terminated.")

	def randPolicy(self, state, actions):
		available_actions = self.actionlist[actions]
		return available_actions[np.random.randint(len(available_actions))]

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


class DP_Agent(RLAgent):
	"""Dynamic Programming (DP) agent."""

	def initRandomPolicy(self):
		"""
		Build the matrix P_pi = sum_a pi(a|s)*Pr_a(s'|s)
		(See tutorial slides 13-15)"""
		Psum = self.world.P.sum(axis=0)
		Pnorm = Psum.sum(axis=1)
		zero_idxs = Pnorm==0.0
		Pnorm[zero_idxs] = 1.0
		self.P_pi = (Psum.T / Pnorm).T

	def evaluatePolicy(self, gamma):
            """
            This method updates the value function with respect to the current policy.
            Note that you don't need to explicitly use / call the policy!
            The agent should have an attribute 'P_pi'...
            INPUTS: 
                gamma: future decay weighting 
            OUTPUTS:
                N/A
            """
            pass
            # HINTS: initialise an update-difference tolerance, 'delta' and check at each
            # iteration if the difference between the new and previous value vectors
            # is below some tolerance...

	def improvePolicy(self):
            """
            This method should improve the policy by making it greedy with respect to the
            value function.
            INPUTS / OUTPUTS: N/A
            """
            pass
            # initialise a 'new' self.P_pi of zeros.
            # for each state:
                # transitions = possible transitions from this state (a property of the 'world')
                # s_next = whichever available transition state has largest value in self.v...
                # make self.P_pi[state,s_next] = 1.0
        
	def policyIteration(self, gamma):
              """
              Iterate between policy evaluation and policy improvement until convergence...
              (or until you get bored. Don't let 'while' loops run unrestricted!)
              INPUTS: 
                    gamma: future decay weighting
              """
              #print("Running policy iteration...")
              pass
              #print("Converged after {} iterations.".fomat(number_of_iterations))

class TDSarsa_Agent(RLAgent):

	def __init__(self, world):
		super(TDSarsa_Agent, self).__init__(world)
		self.policy = self.epsilongreedyQPolicy

	def evaluatePolicyQ(self, gamma, alpha, ntrials):
            """
            This module (should) implement Q-function policy evaluation.
            INPUTS: 
                gamma: future decay weighting 
                alpha:learning rate
                ntrials: number of trials
            """
            delta = 1.0 # after each trial, update this to the 'difference' between the old and new policy. Stop if it gets below some tolerance?
            pass

	def policyIteration(self, gamma, alpha, ntrials):
            """
            Implements policy iteration: flopping between policy evaluation and policy improvement.
            INPUTS: 
                gamma: future decay weighting 
                alpha:learning rate
                ntrials: number of trials
            """
            print("Running TD policy iteration...")
            pass
            #print("Converged after {} iterations.".format(number_of_iterations)


class TDQ_Agent(TDSarsa_Agent):
	""" Q-learning agent. Can learn 'off-policy'"""
	def __init__(self, world):
		super(TDQ_Agent, self).__init__(world)
		self.policy = self.epsilongreedyQPolicy
		self.offpolicy = self.greedyQPolicy

	def choose_offpolicy_action(self):
		state = self.world.get_state()
		actions = self.world.get_actions()
		self.action = self.offpolicy(state, actions)
		return self.action

	def evaluatePolicyQ(self, gamma, alpha, ntrials):
            """
            Evaluate the policy for Q-learning. Remember we must be able to choose
            'off-policy' actions to explore, but we should update with on-policy actions!
            INPUTS: 
                gamma: future decay weighting 
                alpha: learning rate
                ntrials: number of trials
            """
            pass

class TDSarsaLambda_Agent(TDSarsa_Agent):
	
	def __init__(self, world, lamb):
		super(TDSarsaLambda_Agent, self).__init__(world)
		self.policy = self.epsilongreedyQPolicy
		self.lamb = lamb

	def evaluatePolicyQ(self, gamma, alpha, ntrials):
            """
            Evaluate the policy for TD-SARSA-learning. 
            Remember we must store eligibility traces!
            INPUTS: 
                gamma: future decay weighting 
                alpha: learning rate
                ntrials: number of trials
            """
            pass
