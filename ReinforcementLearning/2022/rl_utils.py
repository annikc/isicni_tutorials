

# backward rollout of return value through the reward vector
def discount_rwds(rewards, gamma):
    rewards = np.asarray(rewards)
    disc_rwds = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        running_add = running_add*gamma + rewards[t]
        disc_rwds[t] = running_add
    return disc_rwds
