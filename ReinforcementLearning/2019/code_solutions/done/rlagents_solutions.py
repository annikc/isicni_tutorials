
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
		delta = 1
		maxiters = 1000  # maximum number of iterations
		itr = 0
		while(delta > 0.001 and itr < maxiters):
			itr += 1
			v_new = self.world.R + gamma * self.P_pi.dot(self.v) # note the policy doesn't appear here! It does in pseudocode. but has already been 'applied' to get P_pi
			delta = np.max(np.abs(v_new - self.v))
			self.v = v_new

	def improvePolicy(self):
		self.P_pi = np.zeros((self.world.nstates,self.world.nstates))
		for s in range(self.world.nstates):
			transitions = np.sum(self.world.P[:,s,:],axis=0).astype(bool)
			nextvals = np.full((self.world.nstates,),-1e6)
			nextvals[transitions] = self.v[transitions]
			s_next = np.argmax(nextvals)
			self.P_pi[s,s_next] = 1

	def policyIteration(self, gamma):
		print("Running policy iteration...")
		policyStable = False
		itr = 0
		maxiters = 1000
		while(not policyStable and itr < maxiters):
			itr += 1
			Ppi_old = self.P_pi
			self.evaluatePolicy(gamma)
			self.improvePolicy()
			policyStable = np.array_equal(Ppi_old,self.P_pi)
		print("Converged after %d iterations." % itr)

class TDSarsa_Agent(RLAgent):

	def __init__(self, world):
		super(TDSarsa_Agent, self).__init__(world)
		self.policy = self.epsilongreedyQPolicy

	def evaluatePolicyQ(self, gamma, alpha, ntrials):
		delta = 1.0
		old_q = self.q
		for i in range(ntrials):
			is_terminal = False
			c = 0
			self.reset()
			s = self.state
			a = self.choose_action()
			while not is_terminal:
				c += 1
				is_terminal = self.take_action(a)
				a_prime = self.choose_action() if not is_terminal else 'D'
				self.q[s,self.action_dict[a]] += alpha*(self.reward + gamma*self.q[self.state,self.action_dict[a_prime]] - self.q[s,self.action_dict[a]])
				s = self.state
				a = a_prime
			delta = min(np.max(np.abs(self.q - old_q)),delta)
			old_q = self.q

	def policyIteration(self, gamma, alpha, ntrials):
		print("Running TD policy iteration...")
		policyStable = False
		itr = 0
		maxiters = 1000 # catch the while loop.
		oldA = np.zeros((self.world.nstates,))
		while (not policyStable and itr < maxiters):
			itr += 1
			self.evaluatePolicyQ(gamma, alpha, ntrials)   
			policyStable = np.array_equal(oldA,np.argmax(self.q, axis=1)) # see if policy changes!
			oldA = np.argmax(self.q, axis=1)
		print("Converged after {} iterations.".format(itr))


class TDQ_Agent(TDSarsa_Agent):
	
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
		delta = 1.0
		old_q = self.q
		for i in range(ntrials):
			is_terminal = False
			c = 0
			self.reset()
			s = self.state
			a = self.choose_action()
			while not is_terminal:
				c += 1
				is_terminal = self.take_action(a) # taking an action gives terminality status.
           # explore from the epsilon-greedy policy 
				a_prime = self.choose_action() if not is_terminal else 'D'
           # below we choose a prime from the TRUE policy! 
				a_greedy = self.choose_offpolicy_action() if not is_terminal else 'D'
				self.q[s,self.action_dict[a]] += alpha*(self.reward + gamma*self.q[self.state,self.action_dict[a_greedy]] - self.q[s,self.action_dict[a]])
				s = self.state
				a = a_prime
			delta = min(np.max(np.abs(self.q - old_q)),delta)
			old_q = self.q


class TDSarsaLambda_Agent(TDSarsa_Agent):
	
	def __init__(self, world, lamb):
		super(TDSarsaLambda_Agent, self).__init__(world)
		self.policy = self.epsilongreedyQPolicy
		self.lamb = lamb

	def evaluatePolicyQ(self, gamma, alpha, ntrials):
        delta = 1.0
        old_q = self.q
        eligibility = np.zeros((self.world.nstates,5)) #  store elig traces
        for i in range(ntrials):
            is_terminal = False
            c = 0
            self.reset()
            s = self.state
            a = self.choose_action()
            while not is_terminal:
                c += 1
                is_terminal = self.take_action(a) # this updates other stuff and returs terminality
                a_prime = self.choose_action() if not is_terminal else 'D'
                delta = self.reward + gamma*self.q[self.state,self.action_dict[a_prime]] - self.q[s,self.action_dict[a]]
                eligibility[s,self.action_dict[a]] += 1.0
                self.q[s,self.action_dict[a]] += alpha*(self.reward + gamma*self.q[self.state,self.action_dict[a_prime]] - self.q[s,self.action_dict[a]])
                for s in range(self.world.nstates):
                    for a in range(5):
                        self.q[s,a] += alpha*delta*eligibility[s,a]
                        eligibility[s,a] *= gamma*self.lamb
                        s = self.state
                        a = a_prime
                        delta = min(np.max(np.abs(self.q - old_q)),delta)
                        old_q = self.q