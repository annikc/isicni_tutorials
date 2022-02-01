# testTDPolicyIter.py

from rlagents import TDSarsa_Agent
from gridworld import GridWorldExample3
from plotFunctions import plotStateActionValue, plotGreedyPolicyQ

world = GridWorldExample3()
agent = TDSarsa_Agent(world)

alpha = 0.2
ntrials = 1000
gamma = 0.99
agent.policyIteration(gamma, alpha, ntrials)

plotStateActionValue(agent.q,world)
plotGreedyPolicyQ(agent.q,world)