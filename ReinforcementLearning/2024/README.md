# Tutorial for IBRO-Simons Computational Neuroscience Imbizo 2022
This tutorial is to lay out the major concepts of reinforcement learning 
and to offer some small practical exercises for testing intuition. 

The tutorial covers the following: 
- Markov Decision Processes (MDPs)
- Environments, specifying transition/reward functions
- Agents (basic model-free, value-based agents)

The notebook `RL_tut_exercises.ipynb` leaves small segments code to be completed within pre-written functions. 
There are some exercises that involve full code to be left to the reader. 
Solutions for this code are available in the `RL_tut_solutions.ipynb` notebook. 

## Section 1: Building up to a Markov Decision Process 
Based on David Silver's RL lecture series, we explore a Markov Chain, a Markov Reward Process, and a Markov Decision process. Functions to sample trajectories, collect rewards, and compute returns are defined in `mdp_functions.py`.

## Section 2: Understanding the Environment
In this section we explore specifying the dynamics of the environment with transition and reward functions. We start with a linear track and specify a reward vector and transition dynamics tensor which specifies the probability of moving between states along the track. We then move on to a gridworld environment which follows the same pattern. Gridworld functions are defined in from `env_functions.py`. Here we also write a function to control interaction between an "agent" (with a random policy) and the environment. We explore this random agent in a few different versions of the same environment (open field, obstacles, edge of terminal states). Here also we demonstrate a few different gridworlds from `env_functions.py`. 

## Section 3: Understanding the Agent
We write a general tabular agent class where we define a Q-table to store state-action values. The general tabular agent class also has a `choose_action` method which takes state as an argument and returns an action from the $\epsilon$-greedy policy. This is the parent class for the Monte-Carlo, Q-Learning, and SARSA agents we explore in this section. For each of these types of agents, we define the methods `update_q_table` and `navigate` which have slightly different functionality for each. 

We explore performance of each of these agents in the same 10x10 open field gridworld environment. It is left as an exercise to compare performance of these agents across different types of environments. 

## Miniprojects
Here we outline some projects for learners at different levels of familiarity and comfort with the subject. These projects should take roughly a week of work during afternoon free-work periods at the Imbizo. 