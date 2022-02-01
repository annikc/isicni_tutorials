### annik carson march 2020
import numpy as np


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
    R (list) : reward vector
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

def discount_rwds(rewards, gamma):
    rewards = np.asarray(rewards)
    disc_rwds = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        running_add = running_add*gamma + rewards[t]
        disc_rwds[t] = running_add
    return disc_rwds

# Show computation of return for t= 0
def first_element_return(T,rewards, gamma):
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
    avgd_st_vals = []
    for j in range(len(R)):
        get_state_values = []
        for i in range(num_runs):
            trajectory = []
            rewards = [] 
            done = False
            state_list = ["C1", "C2", "C3", "Pass", "Pub", "FB", "Sleep"]

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
    return avgd_st_vals