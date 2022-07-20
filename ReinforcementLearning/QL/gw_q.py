import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('gym_grid:gridworld-v1')
s = env.reset()
print(env.nstates)
q_table = np.random.uniform(low=-1,high=1, size=(env.nstates, env.action_space.n))

# PARAMETERS
num_episodes = 100000
printfreq    = 1000
learning_rate= 0.1
discount     = 0.98

epsilon      = 0.4
start_eps_decay = 1
end_eps_decay= num_episodes//2
eps_decay_val= epsilon/(end_eps_decay-start_eps_decay)

ep_reward = []
agg_ep_rewards = {'ep':[], 'avg':[], 'min':[], 'max':[]}

for ep in range(num_episodes):
    state = env.reset()
    done = False
    reward_per_episode = 0
    while not done:
        if np.random.random()>epsilon:
            # choose action greedily with respect to state-action values
            action = np.argmax(q_table[state])
        else:
            # produce random action
            action = np.random.randint(0,env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        reward_per_episode += reward
        if not done:
            current_q = q_table[state,action]
            max_future_q = np.max(q_table[new_state])

            new_q = (1-learning_rate)*current_q + learning_rate*(reward + discount*max_future_q)
            q_table[state,action] = new_q
        else:
            q_table[state,action] = 0

        state = new_state
    ep_reward.append(reward_per_episode)

    if end_eps_decay >= ep >= start_eps_decay:
        epsilon -= eps_decay_val

    if ep%printfreq==0:
        average_reward = sum(ep_reward[-printfreq:])/len(ep_reward[-printfreq:])
        agg_ep_rewards['ep'].append(ep)
        agg_ep_rewards['avg'].append(average_reward)
        agg_ep_rewards['min'].append(min(ep_reward[-printfreq:]))
        agg_ep_rewards['max'].append(max(ep_reward[-printfreq:]))
        print(f'Ep {ep}: Avg {average_reward}; Min {min(ep_reward[-printfreq:])}; Max {max(ep_reward[-printfreq:])}')


def pref_Q_action(qtable):
    action_table = np.zeros(env.shape)
    for state in range(env.nstates):
        state2d = env.oneD2twoD(state)
        action_table[state2d] = np.argmax(qtable[state,:])

    return action_table

pref_action = pref_Q_action(q_table)


fig,ax = plt.subplots(2,1,sharex=False)
ax[0].plot(agg_ep_rewards['ep'], agg_ep_rewards['avg'], label='avg')
ax[0].plot(agg_ep_rewards['ep'], agg_ep_rewards['min'], label='min')
ax[0].plot(agg_ep_rewards['ep'], agg_ep_rewards['max'], label='max')
ax[0].legend(loc=0)
a = ax[1].imshow(pref_action, interpolation='none')
cbar = fig.colorbar(a, ax=ax[1],ticks=[0,1,2,3])
cbar.ax.set_yticklabels(['Down','Up','Right','Left'])

plt.show()