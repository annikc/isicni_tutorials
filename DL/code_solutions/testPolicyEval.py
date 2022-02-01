# testPolicyEval.py

from rlagents import DP_Agent
from gridworld import GridWorldExample2
from plotFunctions import plotStateValue

world = GridWorldExample2()
agent = DP_Agent(world)

gamma = 0.99
agent.initRandomPolicy()
agent.evaluatePolicy(gamma)

plotStateValue(agent.v,world)