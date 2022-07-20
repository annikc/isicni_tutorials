import gym 
from gym import spaces, error, utils
import numpy as np
from gym.utils import seeding
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# this class includes all the attributes and functions that define the environment 
# some sections will need to be completed
class BasicEnv(object):
  def __init__(self, env_shape, nactions):
    self.shape    = env_shape                    # tuple describing dimensions of environment (row, column)
    self.nstates  = self.shape[0]*self.shape[1]  # integer number of states 
    self.nactions = nactions                     # integer number of actions available 

    self.T = self.transition_function()          # a tensor of dimension (nstates x nstates x nactions)
    
    self.R = self.reward_function()              # a vector of length nstates - one reward for each state
                                                  # a more complex reward function would be action dependent, 
                                                  # making it a matrix of dimension (nstates x nactions)

  #checks to make sure all the probabilities for a certain action add up to 1
  def check_transition_prob(self, transition_matrix):
    action=0
    check= True
    for action in range(self.nactions):
      for state in range(self.nstates):
        sum=0
        for state2 in range(self.nstates):
          sum+=transition_matrix[state, state2, action]
          if state2==self.nstates-1:
            if "%.*f" % (1,sum)!= "%.*f" % (1,1):
              check=False
              break
    return check

  def transition_function(self):
    # write a function that controls transitions between states

    # initialize a tensor of the desired shape 
    T = np.zeros((self.nstates, self.nstates, self.nactions))

    # the number at position T[s1, s2, a]
    # describes a probability of moving FROM state s1 TO state s2 given an action a 
    
    # check that all transitions probabilities out of one state sum to 1 - 
    #if self.check_transition_prob(T):
    #  return T
    #else:
    #  raise Exception("transition prob does not sum to 1 ") 
    return T

  def reward_function(self):
    # write a function that controls reward
    pass

  def move(self, current_state, selected_action): 
    # write a function that returns a new state and reward for taking a 
    # particular action in a particular state
    
    done=False

    # get next state 
    probabilities = self.T[current_state, : ,selected_action]
    next_state = np.random.choice(np.arange(self.nstates), p=probabilities)
    
    # get reward
    reward=self.R[current_state,0]

    # set condition for when the agent is finished its trial in the environment
    if reward != -1: ### NOTE: This is a poorly defined condition for done -- FIX THIS
      done=True

    return next_state, reward, done

# Specific working Envs
class BoundedGridworld(BasicEnv):
  # this class inherits everything form the BasicEnv class 
  # here we will rewrite the class with specific transition rules which dictate
  # that the agent can move around freely within the grid but moving into a boundary 
  # will have no effect 
  def __init__(self, env_shape, nactions):
    super(BoundedGridworld,self).__init__(env_shape, nactions)        # this calls the init function of BasicEnv so we get self.nstates, etc.

    # here we write over self.T and self.R defined in the BasicEnv.__init__() 
    # using the transition_function and reward_function defined in this class, BoundedGridworld
    self.T = self.grid_transition_bounded()          # a tensor of dimension (nstates x nstates x nactions)
    
    self.R = self.reward_function()              # a vector of length nstates - one reward for each state
                                                 # a more complex reward function would be action dependent, 
                                                 # making it a matrix of dimension (nstates x nactions)
  def grid_transition_bounded(self):
    # initialize
    T = np.zeros((self.nstates, self.nstates, self.nactions))  # down, up, right, left

    # add neighbor connections and jumps, remove for endlines
    #T [starting_state, ending_state, action]
    T[list(range(0, self.nstates-self.shape[1])), list(range(self.shape[1],self.nstates)), 0] = 1     #down
    T[list(range(self.shape[1],self.nstates)),  list(range(0, self.nstates-self.shape[1])),1] = 1     # up
    T[list(range(0, self.nstates-1)),               list(range(1, self.nstates)),              2] = 1     # right
    T[list(range(1, self.nstates)),               list(range(0, self.nstates-1)),              3] = 1     # left

    #remove endlines
    T[list(range(self.shape[1]-1, self.nstates-1, self.shape[1])), list(range(self.shape[1], self.nstates, self.shape[1])),  2 ] = 0    # remove transitions from endlines on right action
    T[list(range(0,self.nstates-self.shape[1]+1, self.shape[1])) , list(range(-1,self.nstates-self.shape[1], self.shape[1])), 3] = 0    # remove transitions from endlines on left action
    # include self transitions 
    #T[start_state, end_state, action]
    T[list(range(self.nstates-self.shape[1],self.nstates)),list(range(self.nstates-self.shape[1], self.nstates)), 0] = 1
    T[list(range(self.shape[1])),list(range(self.shape[1])), 1] = 1    
    T[list(range(self.shape[1]-1, self.nstates, self.shape[1])), list(range(self.shape[1]-1, self.nstates, self.shape[1])),  2 ] = 1
    T[list(range(0,self.nstates-self.shape[1]+1, self.shape[1])) , list(range(0,self.nstates-self.shape[1]+1, self.shape[1])), 3] = 1

    if self.check_transition_prob(T) == True:
      return T
    else:
      raise Exception("transition prob does not sum to 1 ") 
  
  def reward_function(self):
    R = np.ones((self.nstates, 1))*-1 # reward is -1 in all states for all actions
    R[-1] = 100                       # reward is 100 in last state 
    return R 

class EdgewrapGridworld(BasicEnv):
  # this class inherits everything form the BasicEnv class 
  # here we will rewrite the class with specific transition rules which dictate
  # that the agent can move around freely within the grid and that moving off the
  # grid on one side will return you to the other side of the grid (ex. pacman)

  def __init__(self, env_shape, nactions):
    super().__init__(env_shape, nactions)        # this calls the init function of BasicEnv so we get self.nstates, etc.

    # here we write over self.T and self.R defined in the BasicEnv.__init__() 
    # using the grid_transition_edgeWrapping and reward_function defined in this class, EdgewrapGridworld
    self.T = self.grid_transtion_edgeWrapping()
    
    self.R = self.reward_function()
    
  #creates transition matrix for edge-wrapping environment
  def grid_transtion_edgeWrapping(self):    
    # initialize
    T = np.zeros((self.nstates, self.nstates, self.nactions))  #T [starting_state, ending_state, action]

    # add neighbor connections 
    T[list(range(0, self.nstates-self.shape[1])), list(range(self.shape[1],self.nstates)), 0] = 1     #down
    T[list(range(self.shape[1],self.nstates)),  list(range(0, self.nstates-self.shape[1])),1] = 1     # up
    T[list(range(0, self.nstates-1)),               list(range(1, self.nstates)),              2] = 1     # right
    T[list(range(1, self.nstates)),               list(range(0, self.nstates-1)),              3] = 1     # left

    #remove endlines
    T[list(range(self.shape[1]-1, self.nstates-1, self.shape[1])), list(range(self.shape[1], self.nstates, self.shape[1])),  2 ] = 0 # remove transitions from endlines on right action
    T[list(range(0,self.nstates-self.shape[1]+1, self.shape[1])) , list(range(-1,self.nstates-self.shape[1], self.shape[1])), 3] = 0 # remove transitions from endlines on left action
    
    #add in connections to wrap around environment - return you to same line you were in on the other side
    T[list(range(self.nstates-self.shape[1],self.nstates)),list(range(self.shape[1])), 0] = 1
    T[list(range(self.shape[1])),list(range(self.nstates-self.shape[1],self.nstates)), 1] = 1    
    T[list(range(self.shape[1]-1, self.nstates, self.shape[1])),list(range(0,self.nstates-self.shape[1]+1, self.shape[1])) ,  2 ] = 1 
    T[list(range(0,self.nstates-self.shape[1]+1, self.shape[1])) , list(range(self.shape[1]-1, self.nstates, self.shape[1])), 3] = 1
    
    if self.check_transition_prob(T) == True:
      return T
    else:
      raise Exception("transition prob does not sum to 1 ") 
  
  def reward_function(self):
    R = np.ones((self.nstates, 1))*-1 # reward is -1 in all states for all actions
    R[-1] = 100                       # reward is 100 in last state 
    return R 
    
class CliffWorld(BasicEnv):
  def __init__(self, env_shape, nactions):
    super().__init__(env_shape, nactions)        # this calls the init function of BasicEnv so we get self.nstates, etc.

    # here we write over self.T and self.R defined in the BasicEnv.__init__() 
    # using the transition_function and reward_function defined in this class, CliffWorld
    self.T = self.grid_transtion_cliff_bottom()
    
    self.R = self.set_cliff_reward()

  #creates transition vector for cliff environment
  def grid_transtion_cliff_bottom(self):
    T = np.zeros((self.nstates, self.nstates, self.nactions)) #T [starting_state, ending_state, action]

    # add neighbor connections and jumps, remove for endlines
    T[list(range(0, self.nstates-2*self.shape[1])), list(range(self.shape[1],self.nstates-self.shape[1])), 0] = 1     #down
    T[list(range(self.shape[1],self.nstates)),  list(range(0, self.nstates-self.shape[1])),1] = 1     # up
    T[list(range(0, self.nstates-1)),               list(range(1, self.nstates)),              2] = 1     # right
    T[list(range(1, self.nstates)),               list(range(0, self.nstates-1)),              3] = 1     # left

    #remove endlines
    T[list(range(self.shape[1]-1, self.nstates-1, self.shape[1])), list(range(self.shape[1], self.nstates, self.shape[1])),  2 ] = 0    # remove transitions from endlines on right action
    T[list(range(0,self.nstates-self.shape[1]+1, self.shape[1])) , list(range(-1,self.nstates-self.shape[1], self.shape[1])), 3] = 0    # remove transitions from endlines on left action
    # include self transitions 
    #T[start_state, end_state, action]
    T[list(range(self.nstates-self.shape[1],self.nstates)),list(range(self.nstates-self.shape[1], self.nstates)), 0] = 1
    T[list(range(self.shape[1])),list(range(self.shape[1])), 1] = 1    
    T[list(range(self.shape[1]-1, self.nstates, self.shape[1])), list(range(self.shape[1]-1, self.nstates, self.shape[1])),  2 ] = 1
    T[list(range(0,self.nstates-self.shape[1]+1, self.shape[1])) , list(range(0,self.nstates-self.shape[1]+1, self.shape[1])), 3] = 1

    #moving into the cliff states, takes agent back to start 
    T[list(range((self.shape[1]*(self.shape[0]-2))+1,(self.shape[1]*(self.shape[0]-1))-1)), self.nstates-self.shape[1], 0] = 1 #when taking down action from states above cliff states
    T[(self.shape[1]*(self.shape[0]-2)),self.nstates-self.shape[1], 0]=1
    T[(self.shape[1]*(self.shape[0]-1))-1,self.nstates-1, 0]=1
    T[self.nstates-self.shape[1], self.nstates-self.shape[1]+1, 2] = 0
    T[self.nstates-self.shape[1], self.nstates-self.shape[1], 2] = 1 #from the left of cliff states
    T[self.nstates-1, self.nstates-2, 3] = 0
    T[self.nstates-1, self.nstates-self.shape[1], 3] = 1 #from the right of cliff states

    if self.check_transition_prob(T) == True:
      return T
    else:
      raise Exception("transition prob does not sum to 1 ") 
  
  #creates reward vector for the cliff evironment where all cliff states are worth -100 and terminal state is 100
  def set_cliff_reward(self):
    R=np.ones((self.nstates,1))*-1
    R[-1] = 100
    R[list(range((self.shape[1]*(self.shape[0]-2))+1,(self.shape[1]*(self.shape[0]-1))-1))] = -100
    return R
    
##############################################################################################33
class GridWorld(gym.Env):
    metadata = {'render.modes':['human']}
    def __init__(self, rows=20, cols=20, rewards={(5,5):10},
                 env_type=None, rho=0.0, step_penalization=-0.01,
                 terminals=None, obstacles=None, jumps=None,
                 port_shift=None, barheight=None,
                 around_reward=False,
                 **kwargs):
        '''
        :param rows (int): specifying width (x dim) of grid
        :param cols (int): specifying height (y dim) of grid
        :param rewards (dict): reward location coordinates as keys and reward magnitude as values

        :param env_type (str): [None, 'bar','room','tmaze', 'triple_reward']
        :param rho (float): density of obstacles to be randomly positioned in environment
        :param step_penalization (float): penalization for each step (small negative reward)

        :param terminals (list): terminal state coordinates
        :param obstacles (list): obstacle coordinates
        :param jumps (dict): keys and values are coordinates in the environment (x,y):(w,z)
                             where agent can select jump action in coordinate (x,y) and moves to coordinate (w,z)

        :param port_shift (str):  ['equal', 'left', 'right']: type of probability distribution for Kirth's task
        :param barheight (int): height of horizontal line of obstacles dividing the plane

        :param around_reward (bool): whether to randomly choose agent starting location around reward location
        '''
        self.shape = (rows, cols)
        self.c = cols
        self.r = rows

        self.maze_type = env_type
        self.rho = rho

        # basic set up states, rewards
        self.nstates = self.c*self.r
        self.rewards = rewards

        # populate list of obstacles
        self.obstacles_list = obstacles

        if self.maze_type == 'bar':
            if barheight is not None:
                self.barheight 	= barheight
        self.grid, self.useable, self.obstacles_list = self.buildGrid()

        # populate list of terminals
        if terminals is not None:
            if isinstance(terminals, tuple):
                self.terminal2D = [terminals]
                self.terminal = [self.twoD2oneD(terminals)]
            else:
                self.terminal2D = terminals
                self.terminal = [self.twoD2oneD((r,c)) for r,c in terminals]
        else:
            self.terminal2D = []
            self.terminal = []


        # populate list of jump states
        if jumps is not None:
            self.jump = jumps
            self.jump_from = [self.twoD2oneD((r,c)) for r,c in list(jumps.keys())]
            self.jump_to = [self.twoD2oneD((r,c)) for r,c in list(jumps.values())]
        else:
            self.jump = None
            self.jump_from = []
            self.jump_to = []

        # Actions
        self.action_list    = kwargs.get('actionlist', ['Down', 'Up', 'Right', 'Left', 'Jump', 'Poke'])
        self.action_dict    = kwargs.get('actiondict', {x: ind for ind, x in enumerate([x[0] for x in self.action_list])})
        self.action_space   = spaces.Discrete(len(self.action_list))
        self.buildTransitionMatrix()

        # Reward
        self.rwd_action = kwargs.get('rewarded_action', 'Poke')

        self.step_penalization = step_penalization
        self.buildRewardFunction()

        # additional features from specific environment types

        if self.maze_type == 'tmaze':
            if port_shift is not None:
                self.port_shift	= port_shift
            else:
                self.port_shift = 'none'
        if self.maze_type == 'triple_reward':
            self.rwd_loc 		= [(self.r-1, 0), (self.r-1, self.c-1), (0, self.c-1)]
            self.orig_rwd_loc 	= [(self.r-1, 0), (self.r-1, self.c-1), (0, self.c-1)]
            self.starter 		= kwargs.get('t_r_start', (0,0))
            self.start_loc      = self.starter



        # TODO - reset environment including initializing start state
        self.reset()
        self.finish_after_first_reward = True

        self.observation_space = self.get_observation()  # TODO -- make more consistent with openai gym attribute observation_space

    def oneD2twoD(self, idx):
        return (int(idx / self.shape[1]),np.mod(idx,self.shape[1]))

    def twoD2oneD(self, coord_tuple):
        r,c = coord_tuple
        return (r * self.shape[1]) + c

    def buildGrid(self, bound=False): #formerly grid_maker()
        env_types = [None, 'bar','room','tmaze', 'triple_reward']

        if self.maze_type not in env_types:
            raise Exception(f"Environment Type '{self.maze_type}' Not Recognized. \nOptions are: {env_types} \nDefault is Open Field (maze_type = 'none')")

        grid = np.zeros((self.r,self.c), dtype=int)

        # set up obstables for different grid types
        if self.maze_type == 'bar':
            space = 2
            self.rho = 0
            for i in range(self.c-(2*space)):
                grid[self.barheight][i+space] = 1

        elif self.maze_type == 'room':
            self.rho = 0
            vwall = int(self.c/2)
            hwall = int(self.r/2)

            #make walls
            for i in range(self.c):
                grid[vwall][i] = 1
            for i in range(self.r):
                grid[i][hwall] = 1

            # make doors
            self.doors = []
            self.doors.append((vwall, np.random.choice(np.arange(0,vwall))))
            self.doors.append((vwall, np.random.choice(np.arange(vwall+1, self.c))))

            self.doors.append((np.random.choice(np.arange(0,hwall)), hwall))
            self.doors.append((np.random.choice(np.arange(hwall+1, self.r)),hwall))
            for i in self.doors:
                grid[i] = 0


        elif self.maze_type == 'tmaze':
            self.rho = 0
            self.possible_ports = []
            grid = np.ones((self.r, self.c), dtype=int)
            h1, v1 = int(self.c/2), 0
            if h1%2==0:
                for i in range(self.c):
                    grid[v1][i] = 0
                    if i == 0:
                        self.possible_ports.append((i,v1))
                    elif i == self.c-1:
                        self.possible_ports.append((i,v1))
            else:
                for i in range(self.c):
                    grid[v1][i] = 0
                    if i == 0:
                        self.possible_ports.append((i,v1))
                    elif i == self.c-1:
                        self.possible_ports.append((i,v1))

            if self.r > int(self.c/2):
                for i in range(self.r):
                    grid[i][h1] = 0
                    if i == self.r-1:
                        self.possible_ports.append((h1,i))
            else:
                for i in range(self.r):
                    grid[i][h1] = 0
                    if i == self.r-1:
                        self.possible_ports.append((h1,i))

        if self.obstacles_list is None:
            if self.rho != 0:
                maze = np.vstack([[np.random.choice([0,1], p = [1-self.rho, self.rho]) for _ in range(self.c)] for _ in range(self.r)])
                grid = grid + maze
                for reward_loc in self.rewards:
                    if grid[reward_loc[1], reward_loc[0]] == 1:
                        grid[reward_loc[1], reward_loc[0]] = 0

            obstacles = list(zip(np.where(grid==1)[0], np.where(grid==1)[1]))
            self.obstacle2D = obstacles
            self.obstacle = [self.twoD2oneD((r,c)) for r,c in obstacles]

        else:
            for reward_loc in self.rewards:
                if reward_loc in self.obstacles_list:
                    self.obstacles_list.remove(reward_loc)
            if isinstance(self.obstacles_list, tuple):
                self.obstacle2D = [self.obstacles_list]
                self.obstacle = [self.twoD2oneD(self.obstacles_list)]
            else:
                self.obstacle2D = self.obstacles_list
                self.obstacle = [self.twoD2oneD((r,c)) for r,c in self.obstacles_list]
                for coord in self.obstacles_list:
                    grid[coord] = 1

        if bound:
            grid_bound = np.ones((self.r+2, self.c+2), dtype=int)
            grid_bound[1:-1][:,1:-1] = grid
            grid = grid_bound

        # lists of tuples storing locations of open grid space and obstacles (unusable grid space)
        useable_grid = list(zip(np.where(grid==0)[0], np.where(grid==0)[1]))
        obstacles = list(zip(np.where(grid==1)[0], np.where(grid==1)[1]))

        return grid, useable_grid, obstacles

    def buildRewardFunction(self):
        if self.rwd_action in self.action_list:
            self.R = self.step_penalization*np.ones((self.nstates, len(self.action_list)))
            action = self.action_list.index(self.rwd_action)
            for r,c in list(self.rewards.keys()):
                self.R[self.twoD2oneD((r,c)), action] = self.rewards[(r,c)]
        else:
            # specify reward function
            self.R = self.step_penalization*np.ones((self.nstates,))  # rewards received upon leaving state
            for r,c in list(self.rewards.keys()):
                self.R[self.twoD2oneD((r,c))] = self.rewards[(r,c)]

    def buildTransitionMatrix(self):
        # initialize
        self.P = np.zeros((len(self.action_list), self.nstates, self.nstates))  # down, up, right, left, jump, poke

        # add neighbor connections and jumps, remove for endlines
        self.P[0, list(range(0, self.nstates-self.shape[1])), list(range(self.shape[1], self.nstates))] = 1     # down
        self.P[1, list(range(self.shape[1], self.nstates)), list(range(0, self.nstates-self.shape[1]))] = 1  	# up

        self.P[2, list(range(0, self.nstates-1)), list(range(1, self.nstates))] = 1  							# right
        self.P[3, list(range(1, self.nstates)), list(range(0, self.nstates-1))] = 1  							# left
        if len(self.action_list) > 4:
            self.P[4, self.jump_from, self.jump_to] = 1												# jump

        # remove select states
        endlines = list(range(self.shape[1]-1,self.nstates-self.shape[1],self.shape[1]))
        endlines2 = [x+1 for x in endlines]
        self.P[2, endlines, endlines2] = 0	# remove transitions at the end of the grid
        self.P[3, endlines2, endlines] = 0
        for i in range(4):
            self.P[i, :, self.obstacle] = 0  	# remove transitions into obstacles
            self.P[i, self.obstacle, :] = 0  	# remove transitions from obstacles
            self.P[i, self.terminal, :] = 0  	# remove transitions from terminal states
            if len(self.action_list) >4:
                self.P[i, self.jump_from, :] = 0 	# remove neighbor transitions from jump states

        if len(self.action_list) >5:
            # poke should make no transitions between states so everything stays 0
            self.P[5, list(range(0, self.nstates)), list(range(0, self.nstates))] = 1

    def remapTransitionMatrix(self):
        oldP = self.P

        # initalize
        self.P = np.zeros((len(self.action_list), self.nstates, self.nstates))  # down, up, right, left, jump, poke

        for x in range(oldP.shape[0]):
            col = (x + 1) % oldP.shape[0]
            self.P[col, :, :] = oldP[x, :, :]

        print("transition probs remapped")

    def get_random_start_location(self):
        get_start = np.random.choice(len(self.useable))
        start_r = self.useable[get_start][0]
        start_c = self.useable[get_start][1]
        return (start_r, start_c)

    def get_start_location(self, around_reward, **kwargs):
        if around_reward:
            radius = kwargs.get('rad', 5)
            # pick starting location for agent in radius around reward location
            start_buffer = radius  # radius around reward
            starting_reward = list(self.rewards.keys())[np.random.choice(np.arange(len(self.rewards.keys())))]
            get_start_loc = True
            while get_start_loc:
                buf_r = np.random.choice(np.arange(start_buffer))
                start_r = starting_reward[0] + np.random.choice([-1, 1])*buf_r
                if start_r < 0:
                    start_r = 0
                elif start_r > self.grid.shape[0] - 1:
                    start_r = self.grid.shape[0] - 1

                buf_c = np.random.choice(np.arange(start_buffer))
                start_c = starting_reward[1] + np.random.choice([-1, 1])*buf_c
                if start_c < 0:
                    start_c = 0
                elif start_c > self.grid.shape[1] - 1:
                    start_c = self.grid.shape[1] - 1
                if (start_r, start_c) in self.useable:
                    get_start_loc = False
        else: # pick a random starting location for agent within the useable spaces
            get_start = np.random.choice(len(self.useable))
            start_r = self.useable[get_start][0]
            start_c = self.useable[get_start][1]

        start_coord = (start_r, start_c)

        return start_coord

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def get_actions(self):
        slice_n_dice = self.P[:,self.state,:]
        return np.any(slice_n_dice,axis=1)

    def set_reward(self, rewards):
        self.rewards = rewards

        # recalculate reward function
        self.buildRewardFunction()

    def get_reward(self, action):
        action = self.action_list[action][0]
        if len(self.R.shape) > 1:
            reward = self.R[self.state,self.action_dict[action]]
        else:
            reward = self.R[self.state]
        if self.finish_after_first_reward and reward in self.rewards.values():
            self.done = True

        # TODO: fix for kirth
        if self.maze_type == 'tmaze':
            if self.port_shift in ['equal', 'left', 'right']:
                self.shift_rwd(self.port_shift)

        return reward

    def get_observation(self, **kwargs):
        onehot = kwargs.get('onehot', False)
        if onehot:
            obs = np.zeros((1,np.prod(self.grid.shape)))
            obs[0,self.state] = 1
            return np.array([obs])
        else:
            agent_location  = kwargs.get('agtlocation', self.oneD2twoD(self.state))
            reward_location = kwargs.get('rwdlocation', self.rewards)

            #location of reward
            rwd_position = np.zeros_like(self.grid)
            for reward in reward_location:
                rwd_position[reward] = 1

            #location of agent
            agt_position = np.zeros_like(self.grid)
            agt_position[agent_location] = 1

            return np.array([self.grid, rwd_position, agt_position])

    def get_sample_obs(self):
        env_sample = [[],[]]
        for i in self.useable:
            observation = self.get_observation(agtlocation = i)
            env_sample[0].append(observation)
            env_sample[1].append(i)

        return env_sample

    def shift_reward(self,shift): # TODO fix for kirth
        port_rwd_probabilities = [0.333, 0.333, 0.333]
        current_rewarded_port = self.rewards.keys()[0]

        if shift == 'equal':
            dir_prob = [0.5, 0.5]

        elif shift == 'left':
            dir_prob = [0.95, 0.05]

        elif shift == 'right':
            dir_prob = [0.05, 0.95]

        if self.rewards.keys()[0] in self.possible_ports:
            poked_port = self.possible_ports.index(self.rewards.keys()[0])
            right_port = (self.possible_ports.index(self.rewards.keys()[0])+1)%3
            left_port = (self.possible_ports.index(self.rewards.keys()[0])+2)%3

            port_rwd_probabilities[poked_port] = 0
            port_rwd_probabilities[left_port] = dir_prob[0]*port_rwd_probabilities[left_port]
            port_rwd_probabilities[right_port] = dir_prob[1]*port_rwd_probabilities[right_port]

            port_rwd_probabilities = [rp/sum(port_rwd_probabilities) for rp in port_rwd_probabilities]
            new_rwd_loc = np.random.choice(3,1,p=port_rwd_probabilities)
            self.rwd_loc = [self.possible_ports[new_rwd_loc[0]]]
        else:
            print('is not works good')

    #########################
    def reset(self, random_start=True):
        self.random_start = random_start
        if self.random_start:
            self.start = self.get_random_start_location()
        else:
            self.start = self.useable[0]
        self.state = self.twoD2oneD(self.start)

        self.observation = self.get_observation()
        self.done = False

        return self.state

    def step(self, action):
        """
        Args:
            move (str): one of ['D','U','R','L','J'] for down, up, right, left, and jump, respectively.
        Returns:
            tuple (a,b,c): a is the new state, b is the reward value, and c is a bool signifying terminal state
        """

        # check if move is valid, and then move
        x = self.get_actions()
        if not self.get_actions()[action]:
            #raise Exception('Agent has tried an invalid action!')
            pass
        else:
            transition_probs = self.P[action, self.state,:]
            self.state = np.nonzero(transition_probs)[0][0]  # update to new state

        reward = self.get_reward(action) ## self.done is set in this function

        # check if this is a terminal state
        #is_terminal = True if self.state in self.terminal else False
        is_terminal = self.done

        return self.state, reward, is_terminal, {}


class GridWorld4(GridWorld):
    def __init__(self):
        self.action_list = ['Down', 'Up', 'Right', 'Left']
        self.rewarded_action = None
        super().__init__(actionlist=self.action_list, rewarded_action=self.rewarded_action)

