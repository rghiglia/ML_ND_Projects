import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import pandas as pd

import sys
dnm = r'C:\Users\rghiglia\Documents\ML_ND\ML_ND_Projects\smartcab\Version2'
sys.path.append(dnm + r'\smartcab')
sys.path.append(dnm + r'\images')

# Global variables
alpha0 = 0.5
alpha = alpha0
gamma = 0.9


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # TODO: Initialize any additional variables here
        self.acts = ['left', 'right', 'forward', 'stay']
        df_tmp = pd.DataFrame(0*np.ones((2, 4)), index=['red', 'green'], columns=self.acts)
        self.Q = df_tmp
        self.deadline = 0
        

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.deadline = deadline

        # TODO: Update state
        self.state = inputs['light'] # in this case the state is simply defined by the color oof the traffic light at the intersection
        
        # TODO: Select action according to your policy
        # Q-Learner
        act_opt = self.Q.ix[self.state].argmax()
        if act_opt=='stay': act_opt=None
#            print "Current state = %s, qlearner action = %s" % (self.state, act_opt)
        action = act_opt

        # Execute action and get reward
        reward = self.env.act(self, action)


        # TODO: Learn policy based on state, action, reward

        # Maximize Q conditional on the state we are in = S
        q_S = self.Q.ix[self.state] # Q conditional on state S
        q_S_mx = q_S[q_S==q_S.max()] # just in case there are multiple values and the action that maximizes Q(S, a) is not unique
        ix_mx = np.random.randint(0,len(q_S_mx)) # pick one of the equally qlearner actions
        q_S_mx = q_S_mx.iloc[ix_mx]

        # Update Q
        act_tmp = action if action !=None else 'stay'
        global alpha, alpha0, gamma
        alpha *= alpha0
        self.Q.ix[self.state, act_tmp] = \
            (1-alpha)*self.Q.ix[self.state, act_tmp] + \
            alpha*(reward + gamma*q_S_mx) # note: we update the action that has been actually taken, but we update the contribution to Q(S, a) by the action that could have been taken had we behaved qlearnerly
        
#        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
#        print self.Q


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.1)  # reduce update_delay to speed up simulation
    
    nS = 10
    nSb = 10
    succ = np.zeros((nSb,1))
    Succ = np.zeros((nSb,nS))
    for i in range(nS):
        for j in range(nSb):
            sim.run(n_trials=1)  # press Esc or close pygame window to quit
            Succ[i, j] = a.deadline>0
        succ[i] = float(Succ[i,:].sum())/nSb
        print "Q"
        print a.Q
        print Succ[i,:].sum()
        print succ[i]
    


if __name__ == '__main__':
    run()
