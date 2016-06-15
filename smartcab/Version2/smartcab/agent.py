import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
dnm = r'C:\Users\rghiglia\Documents\ML_ND\ML_ND_Projects\smartcab\Version2'
sys.path.append(dnm + r'\smartcab')
sys.path.append(dnm + r'\images')

# Global variables
## Parameter set for random agent without learning
#alpha0 = 0
#alpha = alpha0
#gamma = 0.0
#u0 = 1.0
#q0 = -1.0

## Parameter set #1
#alpha0 = 0.2
#alpha = alpha0
#gamma = 1.0
#u0 = 0
#q0 = 0.0

## Parameter set #1b
#alpha0 = 0.2
#alpha = alpha0
#gamma = 1.0
#u0 = 0.2
#q0 = 0

## Parameter set #1c
#alpha0 = 0.2
#alpha = alpha0
#gamma = 0.5
#u0 = 0.0
#q0 = 1

## Parameter set #2
#alpha0 = 0.95
#alpha = alpha0
#gamma = 0.9
#u0 = 0.10
#q0 = 0

# Parameter set #2
alpha0 = 0.95
alpha = alpha0
gamma = 0.9
u0 = 0
q0 = 0

# Parameter set #2b
alpha0 = 0.95
alpha = alpha0
gamma = 0.95
u0 = 1
uw = 0.5
q0 = -0.5

# Parameter set #2c
alpha0 = 0.95
alpha = alpha0
gamma = 0.5
u0 = 0.5
uw = 0.95
q0 = -0.5


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # TODO: Initialize any additional variables here
        self.acts = ['left', 'right', 'forward', 'stay']
        ix = [clr + '_' + sta for clr in ['red', 'green'] for sta in ['busy', 'free']]
#        df_tmp = pd.DataFrame(0*np.ones((2, 4)), index=['red', 'green'], columns=self.acts)
        global q0
        df_tmp = pd.DataFrame(q0*np.ones((len(ix), 4)), index=ix, columns=self.acts)
        df_cnt = pd.DataFrame(np.zeros((len(ix), 4)), index=ix, columns=self.acts)
        self.Q = df_tmp
        self.Cnt = df_cnt
        self.deadline = 0
        self.alpha = alpha0
        self.av = []
        self.free_red_right = []
        self.free_green_right = []
        self.busy_red_right = []
        self.busy_green_right = []
        

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.alpha = alpha0
        

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.deadline = deadline

        # TODO: Update state
        if (inputs['oncoming']==None) & (inputs['right']==None) & (inputs['left']==None):
            status = 'free'
        else:
            status = 'busy'
        self.state = inputs['light'] + '_' + status # in this case the state is simply defined by the color oof the traffic light at the intersection
        
        # TODO: Select action according to your policy
        # Q-Learner

#        act_opt = self.Q.ix[self.state].argmax()
        q_S = self.Q.ix[self.state] # Q conditional on state S
#        print "q_S",  q_S
#        print "# of actions with max value = {}".format((q_S==q_S.max()).sum())
        q_S_mx = q_S[q_S==q_S.max()] # just in case there are multiple values and the action that maximizes Q(S, a) is not unique
#        print "q_S_mx",  q_S_mx
        ix_mx = np.random.randint(0,len(q_S_mx)) # pick one of the equally valued qlearner actions
#        print "Picking action %s" % q_S_mx.index[ix_mx]
        
        act_opt = q_S_mx.index[ix_mx]
        if act_opt=='stay': act_opt=None
        global u0
        u = random.random()
        u0 *= uw    # a form of simulated annealing
        if u>=u0:
            action = act_opt
        else:
            action = random.choice(['left', 'right', 'forward', None])
        


        # Execute action and get reward
        reward = self.env.act(self, action)

        fig = plt.figure(figsize=(6, 4))
        sns.heatmap(self.Q, annot=True, cmap='YlGnBu')
        plt.title('Q: state = {}, action = {}, reward = {}'.format(self.state, action, reward))
        plt.show()
#        raw_input('Press <ENTER> to continue')


        # TODO: Learn policy based on state, action, reward

        # Maximize Q conditional on the state we are in = S
        q_S = self.Q.ix[self.state] # Q conditional on state S
##        print q_S
##        print "# of actions with max value = {}".format((q_S==q_S.max()).sum())
#        q_S_mx = q_S[q_S==q_S.max()] # just in case there are multiple values and the action that maximizes Q(S, a) is not unique
#        ix_mx = np.random.randint(0,len(q_S_mx)) # pick one of the equally valued qlearner actions
#        q_S_mx = q_S_mx.iloc[ix_mx]
##        print "Picking action %s" % q_S_mx
        q_S_mx = q_S.max() # don't need to handle the tie-break, because I only need the value of the max not the argmax; I need the argmax when I need to choose the action, see code above

        # Store values
        self.av.append(self.alpha)
        self.free_red_right.append(self.Q.ix['red_free','right'])
        self.free_green_right.append(self.Q.ix['green_free','right'])
        self.busy_red_right.append(self.Q.ix['red_busy','right'])
        self.busy_green_right.append(self.Q.ix['green_busy','right'])

        # Update Q
        act_tmp = action if action !=None else 'stay'
        self.Cnt.ix[self.state, act_tmp] += 1 # update count of state_action
        global alpha, alpha0, gamma
        self.alpha *= alpha0
        self.Q.ix[self.state, act_tmp] = \
            (1-self.alpha)*self.Q.ix[self.state, act_tmp] + \
            self.alpha*(reward + gamma*q_S_mx) # note: we update the action that has been actually taken, but we update the contribution to Q(S, a) by the action that could have been taken had we behaved qlearnerly
        
#        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=15.0)  # reduce update_delay to speed up simulation
    
    nE = 3  # epochs
    nR = 10 # trials per epoch
    Succ = np.zeros((nE,nR))
    for i in range(nE):
        for j in range(nR):
            sim.run(n_trials=1)  # press Esc or close pygame window to quit
            Succ[i, j] = a.deadline>0
        print "\nQ"
        print a.Q
        print "\nCnt"
        print a.Cnt
#        print Succ[i,:].sum()
#        print succ[i]
    return (a, Succ)


if __name__ == '__main__':
    a, Succ = run()
    
    # Stats and plots
    print a.Q
    print a.Cnt
    print Succ
    
    fig = plt.figure(figsize=(6, 4))
    plt.plot(a.av)
    plt.title('Alpha')
    
    fig = plt.figure(figsize=(6, 4))
    plt.plot(a.free_red_right, 'r', label='free_red_right')
    plt.plot(a.free_green_right, 'g', label='free_green_right')
    plt.legend(loc='best')
    plt.title('Q')
    
    fig = plt.figure(figsize=(6, 4))
    plt.plot(a.busy_red_right, 'm', label='busy_red_right')
    plt.plot(a.busy_green_right, 'b', label='busy_green_right')
    plt.title('Q')
    plt.legend(loc='best')
    print "\nMean reward free_red_right = %1.2f" % np.array(a.free_red_right).mean()
    print "Mean reward free_green_right = %1.2f" % np.array(a.free_green_right).mean()
    print "\nMean reward busy_red_right = %1.2f" % np.array(a.busy_red_right).mean()
    print "Mean reward busy_green_right = %1.2f" % np.array(a.busy_green_right).mean()
    
    fig = plt.figure(figsize=(6, 4))
    sns.heatmap(a.Cnt/a.Cnt.sum().sum(), annot=True, cmap='YlGnBu')
    plt.title('State-Action Sampling')

    fig = plt.figure(figsize=(6, 4))
    sns.heatmap(a.Q, annot=True, cmap='YlGnBu')
    plt.title('Q')

    # Success rate
    fig = plt.figure(figsize=(6, 4))
    plt.plot(Succ.sum(axis=1)/Succ.shape[1])
    plt.title('Success Rate')
    plt.xlabel('Epoch')
    
