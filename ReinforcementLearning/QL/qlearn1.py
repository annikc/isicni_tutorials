import gym 
import numpy as np
import matplotlib.pyplot as plt

# create environment
env = gym.make("MountainCar-v0")
env.reset()

# print min, max values for obs space
print(f'Min Obs = {env.observation_space.low}, Max Obs = {env.observation_space.high}')
print(f'Num Actions = {env.action_space.n}')

# # turn continuous observation space into discrete by binning
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_window_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
print(discrete_os_window_size)

# make Q table
## TODO why did we pick [-2, 0]?
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
print(q_table.shape)

### PARAMETERS
EPISODES      = 25000
PRINTFREQ     = 2000

LEARNING_RATE = 0.1
DISCOUNT      = 0.95

EPSILON       = 0.5
start_eps_decay = 1
end_eps_decay = EPISODES//2
epsilon_decay_value = EPSILON/(end_eps_decay-start_eps_decay)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_window_size
    transformed_state = tuple(discrete_state.astype(np.int))
    return transformed_state

# test discrete state
state = env.reset()
discrete_state = get_discrete_state(state)
print(f'state {state} --> discrete state {discrete_state}')


ep_reward = []
agg_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

for ep in range(EPISODES):
    if ep%PRINTFREQ==0:
        print(ep)
        render = True
    else:
        render = False

    state = env.reset()
    discrete_state = get_discrete_state(state)
    done = False
    total_reward_per_ep = 0
    while not done:
        if np.random.random()>EPSILON:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0,env.action_space.n)
        new_state, reward, done, _ = env.step(action)
        total_reward_per_ep += reward
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
        if not done:
            # update the table
            current_q    = q_table[discrete_state + (action, )]
            max_future_q = np.max(q_table[new_discrete_state])

            new_q = (1-LEARNING_RATE)*current_q + LEARNING_RATE*(reward + DISCOUNT*max_future_q)
            q_table[discrete_state+(action,)] = new_q

        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    ep_reward.append(total_reward_per_ep)

    if end_eps_decay >= ep >= start_eps_decay:
        EPSILON -= epsilon_decay_value

    if ep%PRINTFREQ==0:
        average_reward = sum(ep_reward[-PRINTFREQ:])/len(ep_reward[-PRINTFREQ:])
        agg_ep_rewards['ep'].append(ep)
        agg_ep_rewards['avg'].append(average_reward)
        agg_ep_rewards['min'].append(min(ep_reward[-PRINTFREQ:]))
        agg_ep_rewards['max'].append(max(ep_reward[-PRINTFREQ:]))
        print(f'EP: {ep} avg: {average_reward}, min: {min(ep_reward[-PRINTFREQ:])}, max: {max(ep_reward[-PRINTFREQ:])}')

env.close()

fig,ax = plt.subplots(2,1,sharex=True)
ax[0].plot(agg_ep_rewards['ep'], agg_ep_rewards['avg'], label='avg')
ax[0].plot(agg_ep_rewards['ep'], agg_ep_rewards['min'], label='min')
ax[0].plot(agg_ep_rewards['ep'], agg_ep_rewards['max'], label='max')
ax[0].legend(loc=0)
plt.show()
