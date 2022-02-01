
import numpy as np
import matplotlib.pyplot as plt

class MountainCarAgent(object):

	def __init__(self, world, tile_shape, ntiles, lamb):
		self.world = world
		self.x = 0.0  # position
		self.v = 0.0  # velocity
		self.x_bounds = (-1.2, 0.6)
		self.v_bounds = (-0.07, 0.07)
		self.tile_shape = tile_shape
		self.ntiles = ntiles
		self.x_tilewidth = (self.x_bounds[1] - self.x_bounds[0])/(tile_shape[0]-1)
		self.v_tilewidth = (self.v_bounds[1] - self.v_bounds[0])/(tile_shape[1]-1)
		self.tile_offsets = np.concatenate([self.x_tilewidth*np.random.rand(ntiles,1),self.v_tilewidth*np.random.rand(ntiles,1)],axis=1)
		self.tile_offsets[0,:] = 0.0
		self.nfeatures = 3 * self.tile_shape[0] * self.tile_shape[1] * self.ntiles
		self.w = np.zeros((self.nfeatures,))
		self.policy = self.epsilonGreedyQPolicy
		self.lamb = lamb

	def get_features(self,pos,vel,action):
		features = np.zeros((self.nfeatures,))
		a = {-1:0,0:1,1:2}[action]
		t0 = a * self.tile_shape[0] * self.tile_shape[1] * self.ntiles
		for i in range(self.ntiles):
			t1 = i * self.tile_shape[0] * self.tile_shape[1]
			t2 = np.int(np.floor((pos-self.x_bounds[0]+self.tile_offsets[i,0])/self.x_tilewidth))
			t3 = np.int(np.floor((vel-self.v_bounds[0]+self.tile_offsets[i,1])/self.v_tilewidth))
			t = t0 + t1 + t2 + t3*self.tile_shape[0]
			features[t] = 1
		return features

	def reset(self):
		self.world.init()
		(self.x, self.v) = self.world.get_state()

	def choose_action(self):
		self.action = self.policy()
		return self.action

	def take_action(self, action):
		((self.x,self.v), self.reward, terminal) = self.world.move(action)
		return terminal

	def run_episode(self, plot=False):
		is_terminal = False
		self.reset()
		c = 0
		while (is_terminal == False):
			c += 1
			action = self.choose_action()
			is_terminal = self.take_action(action)

	def randPolicy(self, epsilon=0):
		return np.random.randint(-1,2)

	def epsilonGreedyQPolicy(self, epsilon=0.0001):
		if np.random.rand() < epsilon:
			return np.random.randint(-1,2)
		else:
			q_vals = np.array([np.dot(self.w, self.get_features(self.x, self.v, -1)),
								np.dot(self.w, self.get_features(self.x, self.v, 0)),
								np.dot(self.w, self.get_features(self.x, self.v, 1))])
			return {0:-1,1:0,2:1}[np.argmax(q_vals)]

	def evaluatePolicyQ(self, gamma, alpha, ntrials):
		eligibility = np.zeros((self.nfeatures,))
		nsteps = np.zeros((ntrials,))
		for i in range(ntrials):  					# number of episodes
			print("Trial nr.",i)
			is_terminal = False
			c = 0
			self.reset()							# initialize state
			x = self.x
			v = self.v
			a = self.choose_action()  	# choose first action
			testx = np.zeros((10001,))
			while not is_terminal:  				# loop through episode
				c += 1
				is_terminal = self.take_action(a)
				a_prime = self.choose_action() if not is_terminal else 0  # arbitrarily choose 0 for terminal state
				q = np.dot(self.w,self.get_features(self.x, self.v, a_prime))
				q_prev = np.dot(self.w,self.get_features(x, v, a))
				delta = self.reward + gamma*q - q_prev
				eligibility += self.get_features(x, v, a)
				self.w += alpha * delta * eligibility
				eligibility *= self.lamb*gamma
				x = self.x
				v = self.v
				a = a_prime
				testx[c] = self.x
				if c>=10000:
					is_terminal=True
					print("Terminated early")
			nsteps[i] = c
			print("Nsteps = ",c)
		return nsteps

