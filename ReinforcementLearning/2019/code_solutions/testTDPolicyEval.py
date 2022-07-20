# testTDPolicyEval.py

from rlagents import TDSarsa_Agent
from gridworld import GridWorldExample3
from plotFunctions import plotStateActionValue

world = GridWorldExample3()
agent = TDSarsa_Agent(world)

gamma = 0.99
alpha = 0.2
ntrials = 500
agent.evaluatePolicyQ(gamma, alpha, ntrials)

plotStateActionValue(agent.q,world)