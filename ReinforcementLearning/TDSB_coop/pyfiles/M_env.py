import numpy as np 

# this class includes all the attributes and functions that define the environment 
class MeetEnv(object):
  def __init__(self, env_shape, nactions, env_type="edge_wrap"):
    self.shape    = env_shape # tuple describing dimensions of environment (row, column)
    self.nstates  = self.shape[0]*self.shape[1]
    self.nactions = nactions
    self.R        = self.set_reward_function()

    if env_type=="edge_wrap":
      self.T        = self.grid_transtion_edgeWrapping()
    elif env_type=="static":
      self.T = self.grid_transtion()
    elif env_type=="linear":
      self.T = self.linear_track_transition()
    elif env_type=="cliff":
      self.T = self.grid_transtion_cliff_bottom()
    else:
      raise Exception("not valid transition type (edge_wrap, static, linear are allowed)")

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
    
  def set_transition_function(self):
    T = np.zeros((self.nstates,self.nstates,self.nactions))
   
    T[0,0,0] = 1
    T[0,1,1] = 0.6
    T[0,2,1] = 0.3
    T[0,3,1] = 0.1
    T[1,0,0] = 1
    T[1,2,1] = 1
    T[2,1,0] = 1
    T[2,3,1] = 1
    T[3,2,0] = 1
    T[3,3,1] = 1
  
    if self.check_transition_prob(T) == True:
      print(self.check_transition_prob(T))
      return T
    else:
      raise Exception("transition prob does not sum to 1 ")

  def linear_track_transition(self):
    T = np.zeros((self.nstates,self.nstates,self.nactions))
    for action in range(self.nactions):
      if action == 0: # left
        for state in range(self.nstates):
          if state == 0:
            T[state,state,action] = 1
          else:
            T[state, state-1, action] = 1 
      elif action == 1: # right
        for state in range(self.nstates):
          if state == self.nstates-1:
            T[state, state, action] = 1
          else:
            T[state, state+1, action] = 1

    if self.check_transition_prob(T) == True:
      return T
    else:
      raise Exception("transition prob does not sum to 1 ") 

  def grid_transtion(self):
    
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

  def grid_transtion_edgeWrapping(self):
    
    # initialize
    T = np.zeros((self.nstates, self.nstates, self.nactions))  # down, up, right, left

    # add neighbor connections and jumps, remove for endlines
    #T [starting_state, ending_state, action]
    T[list(range(0, self.nstates-self.shape[1])), list(range(self.shape[1],self.nstates)), 0] = 1     #down
    T[list(range(self.shape[1],self.nstates)),  list(range(0, self.nstates-self.shape[1])),1] = 1     # up
    T[list(range(0, self.nstates-1)),               list(range(1, self.nstates)),              2] = 1     # right
    T[list(range(1, self.nstates)),               list(range(0, self.nstates-1)),              3] = 1     # left

    #remove endlines
    T[list(range(self.shape[1]-1, self.nstates-1, self.shape[1])), list(range(self.shape[1], self.nstates, self.shape[1])),  2 ] = 0 # remove transitions from endlines on right action
    T[list(range(0,self.nstates-self.shape[1]+1, self.shape[1])) , list(range(-1,self.nstates-self.shape[1], self.shape[1])), 3] = 0 # remove transitions from endlines on left action
    
    #T[start_state, end_state, action]
    T[list(range(self.nstates-self.shape[1],self.nstates)),list(range(self.shape[1])), 0] = 1
    T[list(range(self.shape[1])),list(range(self.nstates-self.shape[1],self.nstates)), 1] = 1    
    T[list(range(self.shape[1]-1, self.nstates, self.shape[1])),list(range(0,self.nstates-self.shape[1]+1, self.shape[1])) ,  2 ] = 1 
    T[list(range(0,self.nstates-self.shape[1]+1, self.shape[1])) , list(range(self.shape[1]-1, self.nstates, self.shape[1])), 3] = 1
    
    if self.check_transition_prob(T) == True:
      return T
    else:
      raise Exception("transition prob does not sum to 1 ") 
  
  def grid_transtion_cliff_bottom(self):

    T = np.zeros((self.nstates, self.nstates, self.nactions))  # down, up, right, left

    # add neighbor connections and jumps, remove for endlines
    #T [starting_state, ending_state, action]
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
    #insert loop
    T[list(range(self.shape[1]*(self.shape[0]-2)+1,(self.shape[1]*(self.shape[0]-1))-2)), self.nstates-self.shape[1], 0] = 1
    T[(self.shape[1]*(self.shape[0]-2)),self.nstates-self.shape[1], 0]=1
    T[(self.shape[1]*(self.shape[0]-1))-1,self.nstates-1, 0]=1
    T[self.nstates-self.shape[1], self.nstates-self.shape[1], 2] = 1
    T[self.nstates-1, self.nstates-2, 3] = 1

    print(T)
    if self.check_transition_prob(T) == True:
      return T
    else:
      raise Exception("transition prob does not sum to 1 ") 
  
  def set_reward_function(self):
    R = np.ones((self.nstates, 1))*-1
    R[-1] = 100
    return R 

  def move(self, current_state, selected_action):
    done=False
    probabilities = self.T[current_state, : ,selected_action]
    next_state = np.random.choice(np.arange(self.nstates), p=probabilities)
    reward=self.R[next_state,0]

    if reward != -1:
      done=True

    return next_state, reward, done

# create environment with cliff 


