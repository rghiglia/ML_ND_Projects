import sys
sys.path.append(r'C:\Users\rghiglia\Documents\ML_ND\smartcab\smartcab')
sys.path.append(r'C:\Users\rghiglia\Documents\ML_ND\smartcab\images')

import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#global alpha
#global cnt
#global strategy
#global time_left
#global time_tot

alpha0 = 0.5
alpha = alpha0
cnt = 1
gamma = 0.9
strategy = 'guided'
sgys = ['stubborn', 'random', 'guided', 'qlearner']

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
        self.A = pd.DataFrame(columns=['action', 'reward', 'cum_rwd', 'light', 'oncoming', 'left', 'right'])
#        print 'Action frame', self.A
        

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

        # Sequence for Q-learning:
        # 0) Gather inputs I
        # 1) Record current state: S
        # 2) Take an action: a
        # 3) Get reward for starting from state S and taking action a: R(S, a); note (S, a) could land you in different states at different times, the consequences of actions are uncertain
        # 4) Now you can evaluate Q_hat(S, a)

        
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator, returns: left, right, forward
        inputs = self.env.sense(self)     # sense returns: {'light': light, 'oncoming': oncoming, 'left': left, 'right': right}
#        deadline = self.env.get_deadline(self)
        # print dir(self.env)
        # print "Deadline", deadline # # time-steps before end
        
        # TODO: Update state
        self.state = inputs['light'] # in this case the state is simply defined by the color oof the traffic light at the intersection

#        # TODO: Select action according to your strategy
        # I am using 'strategy' as a global variable
        if strategy=='stubborn':
            # stubborn action
            action = 'left'
        elif strategy=='random':
            # Random action
            action = random.choice(['left', 'right', 'forward', None])
        elif strategy=='guided':
            # Guided
            action= None
            if inputs['light']=='green':
                action = self.next_waypoint
        elif strategy=='qlearner':
            # qlearner
            act_opt = self.Q[-1].ix[self.state].argmax()
            if act_opt=='stay': act_opt=None
            print "Current state = %s, qlearner action = %s" % (self.state, act_opt)
            action = act_opt
        
        # Execute action and get reward
        reward = self.env.act(self, action) # this will also update the cab to its new location, I think
        print 'Strat %s, action %s, reward %1.2f' % (strategy, action, reward)
        
        # Collect data on action, reward, and inputs
        df_tmp = pd.DataFrame({'action': action, 'reward': reward, 'light': inputs['light'], \
            'oncoming': inputs['oncoming'], 'left': inputs['left'], \
            'right': inputs['right'], 'cum_rwd': 0}, index=[0])
        if len(self.A)==0:
            self.A = df_tmp.copy()
        else:
            self.A = pd.concat([self.A, df_tmp], axis=0, ignore_index=True)
        self.A = self.A[['light', 'oncoming', 'left', 'right', 'action', 'reward', 'cum_rwd']]
        self.A['cum_rwd'] = self.A['reward'].cumsum()
#        print 'Iteration %i, deadline %i' % (cnt, deadline)


        # TODO: Learn strategy based on state, action, reward
        global alpha, alpha0
#        global cnt
#        gamma = 0.9
        print 'Len Q = ', len(self.Q)
        self.Q.append(self.Q[-1].copy()) # store new Q matrix
        print 'Len Q = ', len(self.Q)
        print 'Learn strategy based on state, action, reward: before update'
        print self.Q[0]
        try:
            print self.Q[-2]
        except:
            _ = 1
        print self.Q[-1]
        
        # Maximize Q conditional on the state we are in = S
        q_S = self.Q[-1].ix[self.state] # Q conditional on state S
        q_S_mx = q_S[q_S==q_S.max()] # just in case there are multiple values and the action that maximizes Q(S, a) is not unique
        ix_mx = np.random.randint(0,len(q_S_mx)) # pick one of the equally qlearner actions
        q_S_mx = q_S_mx.iloc[ix_mx]
        
        # Update Q
#        act_tmp = action
#        if act_tmp==None: act_tmp='stay'
        act_tmp = action if action !=None else 'stay'
        alpha *= alpha0
        
        self.Q[-1].ix[self.state, act_tmp] = \
            alpha*self.Q[-1].ix[self.state, act_tmp] + \
            (1-alpha)*(reward + gamma*q_S_mx) # note: we update the action that has been actually taken, but we update the contribution to Q(S, a) by the action that could have been taken had we behaved qlearnerly
        
        print 'Learn strategy based on state, action, reward: after update'
        print 'Len Q = ', len(self.Q)
        print self.Q[0]
        try:
            print self.Q[-2]
        except:
            _ = 1
        print self.Q[-1]

        
#        # Update counter
#        cnt += 1
        
##        print self.env.agent_states
#        for ag in self.env.agent_states:
#            print type(ag)
##            print ag.state['location']
#            print dir(ag)
#            print (ag.get_state())
##            if type(ag)=='__main__.LearningAgent':
##                print ag.location
#        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

#    global alpha
#    global cnt
#    global strategy
    
    # Set up environment and agent

    # Now simulate it
    nS = 1
#    sgys = ['stubborn', 'random', 'guided', 'qlearner']
#    sgys = ['stubborn', 'random', 'guided', 'qlearner']
#    sgys = ['stubborn', 'guided']
    global strategy
    global cnt
    Res = []
    Loc = []
    QQ = []
    clr = ['b', 'g', 'r', 'm', 'y']
    mrk = ['.', '+', '*', '^']
    for i, sgy in enumerate(sgys):
        cnt = 1
        
        # Create environment, agent, and simulator
        e = Environment()  # create environment (also adds some dummy traffic)
        a = e.create_agent(LearningAgent)  # create agent
        e.set_primary_agent(a, enforce_deadline=True)  # set agent to track
        sim = Simulator(e, update_delay=0.011)  # reduce update_delay to speed up simulation
        
        # Assign strategy
        strategy = sgy
        
        # Run simulation with that strategy
        t_tot = []
        t_lft = []
        for iS in range(nS):
            if i==0:
                sim.run(n_trials=1)  # press Esc or close pygame window to quit
                stt = e.stt
                dst = e.dst
            else:
                sim.run(n_trials=1, stt=stt, dst=dst)  # press Esc or close pygame window to quit
                
            t_tot.append(e.time_tot)
            t_lft.append(e.time_lft)
            print "Agent_pst: Start {}, Dest {}".format(e.stt, e.dst)
            
            print "\nActions and rewards, strategy = '%s'" % (strategy)
            Res.append(a.A)
            Loc.append(e.loc)
            QQ.append(a.Q)
    
        # Collect stats
        df_tmp = pd.DataFrame({'tot': t_tot, 'lft': t_lft, 'sgy': strategy})
        df_tmp = df_tmp[['sgy', 'lft', 'tot']]
        if i==0:
            df = df_tmp
        else:
            df = pd.concat([df, df_tmp], axis=0)
        
        df['succ'] = df['lft'].astype(float) / df['tot']
        print df
        
    # Cumulative reward
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for j, res in enumerate(Res):
        ax.plot(res['cum_rwd'], color=clr[j])
    ax.set_title('Cumulative Reward')
    ax.legend(sgys, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.savefig(r'C:\Users\rghiglia\Documents\ML_ND\smartcab\Reward.png', bbox_inches='tight')
    
    # Path
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(stt[0], stt[1], 'r', marker='o', ms=10, markerfacecolor=None, label='Start')
    ax.plot(dst[0], dst[1], 'r', marker='s', ms=10, markerfacecolor=None, label='Stop' )
    for j, loc in enumerate(Loc):
        X = np.zeros((len(loc), 2))
        for k in range(len(loc)):
            X[k, 0], X[k, 1] = loc[k][0], loc[k][1]
        ax.plot(X[:,0]+0.05*j, X[:,1]+0.05*j, ls='-', color=clr[j], label=sgys[j])
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    for j, loc in enumerate(Loc):
        X = np.zeros((len(loc), 2))
        for k in range(len(loc)):
            X[k, 0], X[k, 1] = loc[k][0], loc[k][1]
        ax.plot(X[0,0]+0.05*j, X[0,1]+0.05*j, ls='-', marker='o', color=clr[j])
        ax.plot(X[-1,0]+0.05*j, X[-1,1]+0.05*j, ls='-', marker='x', color=clr[j])
    ax.set_xlim((0, 9))
    ax.set_ylim((0, 7))
    ax.set_title('Path')
    plt.savefig(r'C:\Users\rghiglia\Documents\ML_ND\smartcab\Path.png', bbox_inches='tight')
    
    
    # Value function
    print len(QQ)        # list: one per strategy
    print len(QQ[0])     # list: 42 elements, yeah one per simulation step
    # Q[0][0]           # dataframe
    return QQ
    
    
        
if __name__ == '__main__':
    QQ = run()
#    sns.heatmap(Q[1][0], annot=True, cmap='YlGnBu');
#    sns.heatmap(Q[1][-1], annot=True, cmap='YlGnBu');
    
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    
    nR, nC = Q[0][0].shape
    X, Y = np.meshgrid(range(nR), range(nC))
    for iSgy, sgy in enumerate(sgys):
        Z = QQ[iSgy][-1].T
        print Z
    
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    cset = ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.coolwarm)
#    cset = ax.contourf(X, Y, Z, zdir='x', offset=0, cmap=cm.coolwarm)
##    cset = ax.contourf(X, Y, Z, zdir='y', offset=3, cmap=cm.coolwarm)
#    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.3)
#    
#    ax.set_xticks(np.array(range(nR)))
#    ax.set_xticklabels(Z.columns)
#    ax.set_xlabel('State')
##    ax.set_xlim(-0.5, 2)
#    ax.set_yticks(np.array(range(nC)))
#    ax.set_yticklabels(Z.index)
#    ax.set_ylabel('Action')
##    ax.set_ylim(-1, 4)
#    ax.set_zlabel('Q')
#    ax.set_zlim(-0.5, 2)

#ticksx = np.arange(0.5, 5, 1)
#plt.xticks(ticksx, column_names)
#
#ticksy = np.arange(0.6, 7, 1)
#plt.yticks(ticksy, row_names)

#    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
#    
#    xpos = xpos.flatten()
#    ypos = ypos.flatten()
#    zpos = np.zeros(elements)
    dx = 0.5 * np.ones_like(Z)
    dy = dx.copy()
    dz = Z.values
#    dz = Z.flatten()
    
    sgys = ['stubborn', 'random', 'guided', 'qlearner']
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Z0 = np.zeros_like(X)
    ax.bar3d(X.flatten() - 0.25, Y.flatten(), Z0.flatten(), \
        dx.flatten(), dy.flatten(), dz.flatten(), alpha=0.5)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    plt.xticks(range(nR), Z.columns)
    plt.yticks(range(nC), Z.index)
    ax.set_title('Q: strategy = ' + sgys[iSgy])
    ax.set_xlabel('State')
    ax.set_ylabel('Action')
    ax.set_zlabel('Q')
    # For some reason the Q doesn't seem to change over time ...
