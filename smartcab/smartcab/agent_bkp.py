import sys
sys.path.append(r'C:\Users\rghiglia\Documents\ML_ND\smartcab\smartcab')
sys.path.append(r'C:\Users\rghiglia\Documents\ML_ND\smartcab\images')

import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import pandas as pd


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    # I guess my car is the red one

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.acts = ['left', 'right', 'forward', 'stay']
        df_tmp = pd.DataFrame(np.zeros((2, 4)), index=['red', 'green'], columns=self.acts)
        self.Q = [df_tmp]
        

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.acts = ['left', 'right', 'forward', 'stay']
        df_tmp = pd.DataFrame(np.zeros((2, 4)), index=['red', 'green'], columns=self.acts)
        self.Q = [df_tmp]
        

    def update(self, t):
        
#        Implement the basic driving agent, which processes the following inputs at each time step:
#        
#            Next waypoint location, relative to its current location and heading,
#            Intersection state (traffic light and presence of cars), and,
#            Current deadline value (time steps remaining),
#        
#        And produces some random move/action (None, 'forward', 'left', 'right')
        
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator, returns: left, right, forward
        inputs = self.env.sense(self)     # sense returns: {'light': light, 'oncoming': oncoming, 'left': left, 'right': right}
        # print dir(self.env)
        deadline = self.env.get_deadline(self)
        # print "Deadline", deadline # # time-steps before end
        
        # Sequence:
        # 1) Record state I am in: S
        # 2) Get reward for having landed in that state: R(S)
        # 3) Take an action a for next state
        # 4) Now you can evaluate Q_hat(S, a)

        # TODO: Update state
        self.state = inputs['light']

#        # TODO: Select action according to your policy

#        # Constant action
#        action = 'left'
#        
#        # Random action
#        action = random.choice(['left', 'right', 'forward', None])
#         
#        # Guided
#        action= None
#        if inputs['light']=='green':
#            action = self.next_waypoint

        # Optimal
        act_opt = self.Q[-1].ix[self.state].argmax()
        if act_opt=='stay': act_opt=None
        print "Current state = %s, Optimal action = %s" % (self.state, act_opt)
        action = act_opt
        # Can't do that!
        # This is a STATE NOT an action!
        # Don't I have to survey the possible states then?
        

        # Execute action and get reward
        reward = self.env.act(self, action)


        # TODO: Learn policy based on state, action, reward
        global alpha
        global cnt
        gamma = 1.0/cnt
        self.Q.append(self.Q[-1])
#        print len(self.Q)
#        print self.Q[-1]
        
        # Maximize Q
        q_S = self.Q[-1].ix[self.state] # Q conditional on state S
        q_S_mx = q_S[q_S==q_S.max()] # just in case there are multiple values
        print 'Argmax = ', q_S.argmax()
        print 'Max = '
        print q_S_mx
        ix_mx = np.random.randint(0,len(q_S_mx))
        q_S_mx = q_S_mx.iloc[ix_mx]
        print 'Chosen max = %s = %1.2f' % (self.acts[ix_mx], q_S_mx)
#        print 'Optimal action = ', self.Q[-1].ix[self.state].columns[i_mx]
        
        # Update Q
        act_tmp = action
        if act_tmp==None: act_tmp='stay'
        self.Q[-1].ix[self.state, act_tmp] = reward + gamma*q_S_mx # note: we update the action that has been actually taken, but we update the contribution to Q(S, a) by the action that could have been taken had we behaved optimally
        
        print self.Q[-1]
        

        cnt += 1

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    global alpha
    global cnt
    
    alpha = 0.1
    cnt = 1
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.5)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
