# testQvsSARSA.py

from rlagents import TDSarsa_Agent, TDQ_Agent
from gridworld import CliffWorld
from plotFunctions import plotStateActionValue, plotGreedyPolicyQ

world = CliffWorld()
sarsa_agent = TDSarsa_Agent(world)
alpha = 0.2
ntrials = 2000
gamma = 0.9
sarsa_agent.policyIteration(gamma, alpha, ntrials)
plotStateActionValue(sarsa_agent.q,world)
plotGreedyPolicyQ(sarsa_agent.q,world)

q_agent = TDQ_Agent(world)
alpha = 0.05
ntrials = 2000
gamma = 0.9
q_agent.policyIteration(gamma, alpha, ntrials)
plotStateActionValue(q_agent.q,world)
plotGreedyPolicyQ(q_agent.q,world)