import numpy as np 

def one_hot_rep(num_states, discrete_state_number):
    vector = np.zeros(num_states)
    vector[discrete_state_number] = 1 
    return vector
    
class place_cells(object):
    def __init__(self, env_shape, num_cells, field_size):
        self.env_shape = env_shape
        self.num_cells = num_cells
        self.field_size = field_size
        self.cell_centres = self.get_cell_centres()

    def get_cell_centres(self):
        cell_centres = []
        for i in range(self.num_cells):
            x, y = np.random.random(2)
            cell_centres.append((x,y))
        return np.asarray(cell_centres)

    # define normalized 2D gaussian
    def gaus2d(self, state):
        x  = state[0]
        y  = state[1]
        mx = self.cell_centres[:,0]
        my = self.cell_centres[:,1]
        sx = sy = self.field_size
        return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

    def get_activities(self, states):
        # TODO - array operation
        activities = []
        for state in states:
            xy_state = (state[1]/self.env_shape[1], state[0]/self.env_shape[0]) # input state as (x, y) in [0,1] i.e. portion of total grid area
            place_cell_activity = self.gaus2d(xy_state)
            activities.append(place_cell_activity)
        return np.asarray(activities)