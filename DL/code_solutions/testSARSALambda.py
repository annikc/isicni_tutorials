# testSARSALambda.py

from rlagents import TDSarsaLambda_Agent, TDSarsa_Agent
from gridworld import GridWorldExample3
from plotFunctions import plotStateActionValue, plotGreedyPolicyQ

world = GridWorldExample3()

# 1. test policy evaluation for one trial (episode)
lamb = 0.9
agent = TDSarsaLambda_Agent(world, lamb)
alpha = 0.05
ntrials = 1
gamma = 0.9
#agent.evaluatePolicyQ(gamma, alpha, ntrials)
#plotStateActionValue(agent.q,world)

sarsa0_agent = TDSarsa_Agent(world)
#sarsa0_agent.evaluatePolicyQ(gamma, alpha, ntrials)
#plotStateActionValue(sarsa0_agent.q,world)

# 2. do policy iteration
ntrials = 10
agent.policyIteration(gamma, alpha, ntrials)
plotStateActionValue(agent.q,world)
plotGreedyPolicyQ(agent.q,world)

sarsa0_agent.policyIteration(gamma, alpha, ntrials)
plotStateActionValue(sarsa0_agent.q,world)
plotGreedyPolicyQ(agent.q,world)