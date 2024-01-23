### annik carson jan 2024
import numpy as np
import pandas as pd

# backward rollout of return value through the reward vector
def discount_rwds(rewards, gamma):
    rewards = np.asarray(rewards)
    disc_rwds = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        running_add = running_add*gamma + rewards[t]
        disc_rwds[t] = running_add
    return disc_rwds

def sample_MC_trajectory(S, P):
    '''
    S (list) : state indicies
    P (array): transition probability matrix
    '''
    trajectory = []
    done = False
    state_list = ["C1", "C2", "C3", "Pass", "Pub", "FB", "Sleep"]
    terminal_states = ["Sleep", "Pass"]
    terminal_indices = [state_list.index(x) for x in terminal_states]
    
    # pick first state randomly (not including terminal states)
    state = np.random.choice(np.delete(S, terminal_indices))
    trajectory.append(state_list[state])
    
    while not done:
        # get next states from distribution given by
        # the row of P corresponding to current state
        state = np.random.choice(S, 1, p = P[state])[0]
        trajectory.append(state_list[state])
        # if in terminal state, finish trajectory
        if state == state_list.index("Sleep"): 
            done = True

    return trajectory

def sample_MRP_trajectory(S, P, R):
    '''
    S (list) : state indicies
    P (array): transition probability matrix
    R (array) : reward vector
    '''

    trajectory = []
    rewards = [] 
    done = False
    state_list = ["C1", "C2", "C3", "Pass", "Pub", "FB", "Sleep"]

    terminal_states = ["Sleep", "Pass"]
    terminal_indices = [state_list.index(x) for x in terminal_states]
    
    # pick first state randomly (not including terminal states)
    state = np.random.choice(np.delete(S, terminal_indices))
    trajectory.append(state_list[state])
    rewards.append(R[state])
    while not done:
        # get next states from distribution given by
        # the row of P corresponding to current state
        state = np.random.choice(S, 1, p = P[state])[0]
        trajectory.append(state_list[state])
        rewards.append(R[state])
        # if in terminal state, finish trajectory
        if state == state_list.index("Sleep"): 
            done = True

    return trajectory, rewards

def sample_MDP_trajectory(S, P, R, start_state, action_sequence=None):
    '''
    P (array)              : transition probability matrix
    R (list)               : reward vector
    start_state (str)      : state in the state list
    action_sequence (list) : list of numbers corresponding to actions taken at each step 
    '''

    trajectory = []
    rewards = [] 
    done = False
    state_list = ["C1", "C2", "C3", "Pass", "Pub", "FB", "Sleep"]
    action_list = ['Chill', 'Study']
    # check state is valid 
    if not start_state in state_list:
        raise Exception('Agent has tried an invalid action!')
    
    # add starting state to trajectory (not including terminal states)
    state = state_list.index(start_state)
    trajectory.append(state_list[state])
    rewards.append(R[state])
    
    if action_sequence is not None:
        action_seq = []
        for action in action_sequence:
            if not done:
                # get next states from distribution given by
                # the row of P corresponding to current state
                next_state = np.random.choice(S, 1, p = P[action,state])[0]
                trajectory.append(state_list[next_state])
                action_seq.append(action_list[action])
                rewards.append(R[next_state])
                
                print(state_list[state],action,state_list[next_state],R[next_state])
                # update the information carried in the state variable
                state = next_state

                # if in terminal state, finish trajectory
                if state == state_list.index("Sleep"): 
                    done = True
    
        
        # after taking all of the steps, append one more empty action to make the shapes the same
        # there is no "next action" after you finish stepping through your action_sequence
        action_seq.append("None")

    else:
        ## use a random policy for action selection
        action_seq = []
        while not done:
            action = np.random.choice([0,1])
            state = np.random.choice(S, 1, p = P[action,state])[0]
            trajectory.append(state_list[state])
            action_seq.append(action_list[action])
            rewards.append(R[state])
            # if in terminal state, finish trajectory
            if state == state_list.index("Sleep"): 
                done = True
                # after taking all of the steps, append one more empty action to make the shapes the same
                # once in a terminal state you don't take another action
                action_seq.append("None")

    return trajectory, action_seq, rewards

# Show computation of return for first state in trajectory (eg. t=0)
def first_element_return(T, rewards, gamma):
    '''
    T (list)        : trajectory of states visited
    rewards (array) : rewards received along trajectory
    gamma (float)   : discount factor
    '''
    discount_powers = []
    for i in range(len(T)):
        discount_powers.append(gamma**i)
    
    comp1 = f"({discount_powers[0]})({rewards[0]})"
    for i in range(len(T)-1):
        comp1 += f" + ({discount_powers[i+1]})({rewards[i+1]})"
    print(f"\nG_0 ({T[0]} at t=0):\n={comp1}")
    
    comp2 = f"{discount_powers[0] * rewards[0]}"
    for i in range(len(T)-1):
        comp2 += f" + {discount_powers[i+1] * rewards[i+1]}"
    
    print(f"= {comp2}")
    
    comp3 = sum([x*y for x,y in zip(discount_powers, rewards)])
    print(f"= {comp3}")

def get_MRP_values(S, P, R, gamma, num_runs):
    '''
    S (list) : state indicies
    P (array): transition probability matrix
    R (list) : reward vector
    '''
    state_list = ["C1", "C2", "C3", "Pass", "Pub", "FB", "Sleep"]
    avgd_st_vals = []
    for j in range(len(state_list)):
        get_state_values = []
        for i in range(num_runs):
            trajectory = []
            rewards = [] 
            done = False

            # start all of your trajectories in jth state 
            state = j
            trajectory.append(state_list[state])
            rewards.append(R[state])
            while not done:
                # get next states from distribution given by
                # the row of P corresponding to current state
                state = np.random.choice(S, 1, p = P[state])[0]
                trajectory.append(state_list[state])
                rewards.append(R[state])
                # if in terminal state, finish trajectory
                if state == state_list.index("Sleep"): 
                    done = True

            get_state_values.append(discount_rwds(rewards, gamma)[0])
        avgd_st_vals.append(np.round(np.mean(get_state_values),4))
    avgd_st_vals_dict = dict(zip(state_list,avgd_st_vals))
    return avgd_st_vals_dict


def show_trajectory_table(T, rewards, gammas, actions=None):
    '''
    T (list)        : states visited along trajectory
    rewards (array) : rewards collected along trajectory
    gammas (list)   : discount factors to compute return values
    '''
    if actions==None:
        data = {'state':T, 'reward':rewards}
        headers = ["Step","Reward"]
    else:
        data = {'state':T,'actions':actions, 'reward':rewards}
    # calculate returns for each reward 
    for gamma in gammas:
        G = discount_rwds(rewards, gamma)
        data[f'G ({chr(947)}={gamma})']=G

    table = pd.DataFrame(data)
    return table

def state_rewards(s, R=[-2., -2., -2., 10., 1., -1., 0.]):
   # where indices of R correspond to ["C1", "C2", "C3", "Pass", "Pub", "FB", "Sleep"]
   return R[s]

#def state_action_reward(s,a):
#    R = np.zeros((n_states, n_actions))
#    # to do: write a matrix which specifies how much reward the agent gets for each action in each state
#    
#    # return the value of the reward the agent gets when it selection action=a while in state=s
#    pass 
