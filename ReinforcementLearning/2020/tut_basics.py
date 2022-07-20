import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import cm
import matplotlib.colorbar as colorbar
import matplotlib.colors as colors
## --------------------------------------------------------------------------------
## MULTIARMED BANDIT
## --------------------------------------------------------------------------------
def bandit1(): 
    return np.random.choice([0,1]) 

def bandit2():
    return np.random.choice([1,10], p=[0.7,0.3])

def bandit3():
    return np.random.choice([1,10],p=[0.9,0.1])

## --------------------------------------------------------------------------------
## GRIDWORLD 
## --------------------------------------------------------------------------------
def run_episode(agent, print_details=False):
    print("Running episode...")
    is_terminal = False
    agent.reset()
    c = 0  # step counter
    while (is_terminal == False):
        c += 1 # increase step counter
        prev_state = oneD2twoD(agent.state,agent.world.shape)
        action = agent.choose_action()
        is_terminal = agent.take_action(action)
        state = oneD2twoD(agent.state,agent.world.shape)
        if print_details:
            print("Step %d: move from (%d,%d) to (%d,%d), reward = %.2f" % (c,prev_state[0],prev_state[1],state[0],state[1],agent.reward))
    print(f"Terminated after {c} episodes.")


def oneD2twoD(idx, shape):
	return (int(idx / shape[1]), np.mod(idx,shape[1]))

def twoD2oneD(r, c, shape):
	return r * shape[1] + c

class GridWorld(object):
	
	action_dict = {'D':0, 'U':1, 'R':2, 'L':3, 'J':4}

	def __init__(self, shape, start, terminals, obstacles, rewards, jumps):
		"""
		Args:
			shape (tuple): defines the shape of the gridworld
			start (tuple): defines the starting position of the agent
			terminals (tuple or list): defines terminal states (end of episodes)
			obstacles (tuple or list): defines obstacle squares (cannot be visited)
			rewards (dictionary): states to reward values for EXITING those states
			jumps (dictionary): non-neighbor state to state transitions
		"""
		self.shape = shape
		self.nstates = shape[0]*shape[1]
		self.start = twoD2oneD(start[0], start[1], shape)
		self.state = 0
		if isinstance(terminals, tuple):
			self.terminal2D = [terminals]
			self.terminal = [twoD2oneD(terminals[0],terminals[1],shape)]
		else:
			self.terminal2D = terminals
			self.terminal = [twoD2oneD(r,c,shape) for r,c in terminals]
		if isinstance(obstacles, tuple):
			self.obstacle2D = [obstacles]
			self.obstacle = [twoD2oneD(obstacles[0],obstacles[1],shape)]
		else:
			self.obstacle2D = obstacles
			self.obstacle = [twoD2oneD(r,c,shape) for r,c in obstacles]
		self.jump = jumps
		self.jump_from = [twoD2oneD(x,y,shape) for x,y in list(jumps.keys())]
		self.jump_to = [twoD2oneD(x,y,shape) for x,y in list(jumps.values())]
		self.buildTransitionMatrices()

		self.R = np.zeros((self.nstates,))  # rewards received upon leaving state
		for r,c in list(rewards.keys()):
			self.R[twoD2oneD(r,c,self.shape)] = rewards[(r,c)]

	def buildTransitionMatrices(self):

		# initialize
		self.P = np.zeros((5, self.nstates, self.nstates))  # down, up, right, left, jump

		# add neighbor connections and jumps, remove for endlines
		self.P[0,list(range(0,self.nstates-self.shape[1])),list(range(self.shape[1],self.nstates))] = 1;  	# down
		self.P[1,list(range(self.shape[1],self.nstates)),list(range(0,self.nstates-self.shape[1]))] = 1;  	# up
		self.P[2,list(range(0,self.nstates-1)),list(range(1,self.nstates))] = 1  							# right
		self.P[3,list(range(1,self.nstates)),list(range(0,self.nstates-1))] = 1  							# left
		self.P[4,self.jump_from,self.jump_to] = 1												# jump

		# remove select states
		endlines = list(range(self.shape[1]-1,self.nstates-self.shape[1],self.shape[1]))
		endlines2 = [x+1 for x in endlines]
		self.P[2,endlines,endlines2] = 0	# remove transitions at the end of the grid
		self.P[3,endlines2,endlines] = 0
		for i in range(4):
			self.P[i,:,self.obstacle] = 0  	# remove transitions into obstacles
			self.P[i,self.obstacle,:] = 0  	# remove transitions from obstacles
			self.P[i,self.terminal,:] = 0  	# remove transitions from terminal states
			self.P[i,self.jump_from,:] = 0 	# remove neighbor transitions from jump states 

	def init(self):
		self.state = self.start

	def get_state(self):
		return self.state

	def set_state(self, state):
		self.state = state

	def get_actions(self):
		return np.any(self.P[:,self.state,:],axis=1)

	def move(self, action):
		"""
		Args:
			move (str): one of ['D','U','R','L','J'] for down, up, right, left, and jump, respectively.
		Returns:
			tuple (a,b,c): a is the new state, b is the reward value, and c is a bool signifying terminal state
		"""
		
		# check if move is valid, and then move
		if not self.get_actions()[self.action_dict[action]]:
			raise Exception('Agent has tried an invalid action!')
		reward = self.R[self.state]
		self.state = np.nonzero(self.P[self.action_dict[action],self.state,:])[0][0]  # update to new state

		# check if this is a terminal state
		is_terminal = True if self.state in self.terminal else False

		return (self.state, reward, is_terminal)

class GridWorldExample1(GridWorld):
    def __init__(self):
        shape = (4,5)
        start = (3,0)
        terminals = (3,4)
        obstacles = [(0,0),(2,0),(3,2),(2,2),(1,2),(1,4),(2,4)]
        rewards = {(3,3):1,(2,4):1}
        jumps = {(0,4):(3,0)}
        super(GridWorldExample1, self).__init__(shape, start, terminals, obstacles, rewards, jumps)

class GridWorldExample2(GridWorld):
    def __init__(self):
        shape = (5,7)
        start = (4,0)
        terminals = (4,6)
        obstacles = [(1,1),(2,1),(3,1),(4,1),(0,3),(1,3),(2,3),(3,3),(2,5),(3,5),(4,5)]
        rewards = {(3,6):10}
        jumps = {(0,5):(4,0)}
        super(GridWorldExample2, self).__init__(shape, start, terminals, obstacles, rewards, jumps)

class GridWorldExample3(GridWorld):
    def __init__(self):
        shape = (5,8)
        start = (0,0)
        terminals = (0,7)
        obstacles = [(0,1),(1,1),(2,1),(3,1),(3,3),(3,4),(0,5),(1,5),(2,5),(3,5),(3,7)]
        rewards = {} 
        for i in [0,2,3,4,8,10,11,12,14,15,16,18,19,20,22,23,24,26,30,32,33,34,35,36,37,38,39]:
            rewards[oneD2twoD(i,shape)] = -1
        jumps = {(2,7):(0,4),(1,7):(2,4)}
        super(GridWorldExample3, self).__init__(shape, start, terminals, obstacles, rewards, jumps)

class CliffWorld(GridWorld):
    def __init__(self):
        shape = (3,7)
        start = (2,0)
        terminals = (2,6)
        obstacles = []
        rewards = {}
        jumps = {}
        for i in range(14):
            rewards[oneD2twoD(i,shape)] = -1
        rewards[(2,0)]=-1
        for i in range(15,21):
            rewards[oneD2twoD(i,shape)] = -10
            jumps[oneD2twoD(i,shape)] = (2,0)
        super(CliffWorld, self).__init__(shape, start, terminals, obstacles, rewards, jumps)









































class env(object):
	def __init__(self, shape, start, terminals, obstacles, jumps, actions, rewards, **kwargs):
		"""
		Args:
			shape (tuple): defines the shape of the gridworld
			start (tuple): defines the starting position of the agent
			terminals (tuple or list): defines terminal states (end of episodes)
			obstacles (tuple or list): defines obstacle squares (cannot be visited)
			rewards (dictionary): states to reward values for EXITING those states
			jumps (dictionary): non-neighbor state to state transitions
		"""
		self.shape = shape
		self.rows = shape[0]
		self.cols = shape[1]
		self.strt = start
		self.term = terminals
		self.obst = obstacles
		self.jump = jumps
		self.acts = actions
		self.rwds = rewards

		self.free = []
		for i in range(self.rows):
		    for j in range(self.cols):
		        if (i,j) in self.obst:
		            pass
		        else:
		            self.free.append((i,j))
		self.state = self.strt

		self.nstates = self.rows*self.cols
		self.jump_from = [twoD2oneD(x,y,shape) for x,y in list(jumps.keys())]
		self.jump_to = [twoD2oneD(x,y,shape) for x,y in list(jumps.values())]

		self.R = np.zeros((self.nstates,))  # rewards received upon leaving state
		for r,c in list(rewards.keys()):
			self.R[twoD2oneD(r,c,self.shape)] = rewards[(r,c)]

	def move(self, state, action):
		x = state[0]
		y = state[1]
		if action not in self.acts: 
			print('Invalid action selection')
		if action == 'down':
			x = x
			if y <self.rows-1:
				y = y + 1
			else:
				y = y
		if action == 'up':
			x = x
			if y > 0:
				y = y - 1
			else:
				y = y
		if action == 'right':
			y = y
			if x<self.cols-1:
				x = x+1
			else:
				x = x
		if action == 'left':
			y = y
			if x > 0:
				x = x-1

		if state in self.jump and action == 'jump':
			x = self.jump[state][0]
			y = self.jump[state][1]

		self.state = (x,y)
		return self.state

	def reward(self, state):
		if state in self.rwds:
			return self.rwds[state]
		else:
			return 0 

	def transition_matrix(self):
		# initialize empty transition matrix 
		self.P = np.zeros((len(self.acts), self.nstates, self.nstates))
		for ind, action in enumerate(self.acts[:-1]):
			for cur_state in self.free:
				next_state = self.move(cur_state, action)
				state_from_ind = twoD2oneD(cur_state[0],cur_state[1],self.shape)
				state_to_ind = twoD2oneD(next_state[0],cur_state[1],self.shape)

				self.P[ind, state_from_ind, state_to_ind] = 1



## plot environment for gridworld problem 
def plot_grid(env, **kwargs):
    # optional argument to save figure you create
    save = kwargs.get('save', False)
    
    # make figure object
    fig = plt.figure()
    ax = fig.gca()
    cmap = cm.get_cmap('Set1')

    # get number of rows and columns from the shape of the grid (passed to function)
    n_rows = env.rows
    n_cols = env.cols

    # get info about other aspects of environment
    obst = env.obst
    start_loc = env.strt
    terminal_states = env.term
    jumps = env.jump

    # make grid environment
    grid = np.zeros((n_rows, n_cols))
    # populate environments with obstacles
    for i in obst: 
        grid[i] = 1 

    # show the basic grid    
    plt.imshow(grid, cmap ='bone_r')
    # add gridlines for visibility
    plt.vlines(np.arange(n_cols)+.5, ymin=0-.5, ymax = n_rows-.5, color='k', alpha=0.2)
    plt.hlines(np.arange(n_rows)+.5, xmin=0-.5, xmax = n_cols-.5, color='k', alpha=0.2)
    
    # add annotation for start location
    plt.annotate('S', (start_loc[1], start_loc[0]))
    
    # add annotations for terminal location/s
    for i in terminal_states:
        ax.add_patch(patches.Rectangle(np.add((i[1],i[0]),(-.5,-.5)), 1,1, lw = 3, ec = 'k', color = 'gray', alpha=0.5))
        plt.annotate('T', (i[1],i[0]))

    # add annotations for jump states
    for ind, i in enumerate(jumps.items()):
        start_jump = i[0]
        end_jump = i[1]
        colour = cmap(ind)        
        ax.add_patch(patches.Rectangle(np.add((start_jump[1],start_jump[0]),(-.5,-.5)), 1,1, fill=False, ec = colour, lw = 2, ls = "--"))
        ax.add_patch(patches.Rectangle(np.add((end_jump[1],end_jump[0]),(-.5,-.5)), 1,1, color = colour ,alpha=0.5))

    # statement for saving if optional arg save==True
    if save: 
        plt.savefig('./gridworld.png', format='png')

# value plots for gridworld env
def plot_valmap(env, value_array, save=False, **kwargs):
	'''
	:param maze: the environment object
	:param value_array: array of state values
	:param save: bool. save figure in current directory
	:return: None
	'''
	title = kwargs.get('title', 'State Value Estimates')
	vals = value_array.copy()

	fig = plt.figure()
	ax1 = fig.add_axes([0, 0, 0.85, 0.85])
	axc = fig.add_axes([0.75, 0, 0.05, 0.85])
	vmin, vmax = kwargs.get('v_range', [np.min(value_array), np.max(value_array)])

	cmap = plt.cm.Spectral_r
	cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
	for i in env.obst:
		vals[i[0], i[1]] = np.nan
	
	cb1 = colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm)
	ax1.imshow(vals, cmap=cmap, vmin = vmin, vmax = vmax)

	ax1.set_aspect('equal')
	plt.title(title)
	plt.show()


# policy plots for gridworld env
def make_arrows(action, probability):
	'''
	:param action:
	:param probability:
	:return:
	'''
	if probability == 0:
		dx, dy = 0, 0
		head_w, head_l = 0, 0
	else:
		if action == 0:  # N
			dx = 0
			dy = -.25
		elif action == 1:  # E
			dx = .25
			dy = 0
		elif action == 2:  # W
			dx = -.25
			dy = 0
		elif action == 3:  # S
			dx = 0
			dy = .25
		elif action == 4:  # stay
			dx = -.1
			dy = -.1
		elif action == 5:  # poke
			dx = .1
			dy = .1

		head_w, head_l = 0.1, 0.1

	return dx, dy, head_w, head_l

