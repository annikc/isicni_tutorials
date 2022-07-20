
import numpy as np
from mountainworld import MountainWorld
from mountaincar_agent import MountainCarAgent
import matplotlib.pyplot as plt

world = MountainWorld()

# parameters
lamb = 0.9
gam = 0.99
alp = 0.001

# NOTE: this may take a few minutes to run
car = MountainCarAgent(world, (20,5), 3, lamb)
nsteps = car.evaluatePolicyQ(gam, alp, 200)

# plot performance over time
plt.plot(nsteps)
plt.title("Mountain Car Performance")
plt.ylabel("Number of time steps to goal")
plt.xlabel("Episode nr.")
plt.show()