import sys
dnm = r'C:\Users\rghiglia\Documents\ML_ND\ML_ND_Projects'
sys.path.append(dnm + r'\smartcab\smartcab')
sys.path.append(dnm + r'\smartcab\images')

import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

global stt0, dst0, sgys

alpha0 = 0.5
alpha = alpha0
cnt = 1
gamma = 0.9
strategy = 'guided'
#sgys = ['stubborn', 'random', 'guided', 'qlearner']
sgys = ['guided', 'qlearner']
stt0 = (2, 2)
dst0 = (6, 5)

global Q_est
Q_est = pd.DataFrame(1*np.ones((2, 4)), index=['red', 'green'], columns=['left', 'right', 'forward', 'stay'])

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    # I guess my car is the red one

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # TODO: Initialize any additional variables here
        self.acts = ['left', 'right', 'forward', 'stay']
#        df_tmp = pd.DataFrame(np.zeros((2, 4)), index=['red', 'green'], columns=self.acts)
        df_tmp = pd.DataFrame(1*np.ones((2, 4)), index=['red', 'green'], columns=self.acts)
        global Q_est
        df_tmp = Q_est
        
        self.Q = [df_tmp]
        self.A = pd.DataFrame(columns=['action', 'reward', 'cum_rwd', 'light', 'oncoming', 'left', 'right'])
#        print 'Action frame', self.A
        

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.acts = ['left', 'right', 'forward', 'stay']
#        df_tmp = pd.DataFrame(np.zeros((2, 4)), index=['red', 'green'], columns=self.acts)
        df_tmp = pd.DataFrame(1*np.ones((2, 4)), index=['red', 'green'], columns=self.acts)
        global Q_est
        df_tmp = Q_est
        
        self.Q = [df_tmp]
        self.A = pd.DataFrame(columns=['action', 'reward', 'cum_rwd', 'light', 'oncoming', 'left', 'right'])
        

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
        inputs = self.env.sense(self)     # sense() returns: {'light': light, 'oncoming': oncoming, 'left': left, 'right': right}
        
        # TODO: Update state
        self.state = inputs['light'] # in this case the state is simply defined by the color oof the traffic light at the intersection

#        # TODO: Select action according to your strategy
        # I am using 'strategy' as a global variable
        if strategy=='stubborn':
            # Stubborn action
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
            # Q-Learner
            act_opt = self.Q[-1].ix[self.state].argmax()
            if act_opt=='stay': act_opt=None
#            print "Current state = %s, qlearner action = %s" % (self.state, act_opt)
            action = act_opt
        
        # Execute action and get reward
        reward = self.env.act(self, action) # this will also update the cab to its new location, I think
#        print 'Strat %s, action %s, reward %1.2f' % (strategy, action, reward)
        
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
        self.Q.append(self.Q[-1].copy()) # store new Q matrix
        
        # Maximize Q conditional on the state we are in = S
        q_S = self.Q[-1].ix[self.state] # Q conditional on state S
        q_S_mx = q_S[q_S==q_S.max()] # just in case there are multiple values and the action that maximizes Q(S, a) is not unique
        ix_mx = np.random.randint(0,len(q_S_mx)) # pick one of the equally qlearner actions
        q_S_mx = q_S_mx.iloc[ix_mx]
        
        # Update Q
        act_tmp = action if action !=None else 'stay'
        alpha *= alpha0
        self.Q[-1].ix[self.state, act_tmp] = \
            alpha*self.Q[-1].ix[self.state, act_tmp] + \
            (1-alpha)*(reward + gamma*q_S_mx) # note: we update the action that has been actually taken, but we update the contribution to Q(S, a) by the action that could have been taken had we behaved qlearnerly
        
#        # Update counter
#        cnt += 1
        
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    global strategy
    global cnt
    global sgys
    
    R4, L4, Q4 = [], [], []
    clr = ['b', 'g', 'r', 'm', 'y']
    mrk = ['.', '+', '*', '^']
    
    # Set up environment and agent

    # Now simulate it
    nS = 10
    T_tot = pd.DataFrame({sgy: np.zeros((nS, 1))[:,0] for sgy in sgys})
    T_lft = pd.DataFrame({sgy: np.zeros((nS, 1))[:,0] for sgy in sgys})
#    sgys = ['stubborn', 'random', 'guided', 'qlearner']
    
    # Need to change the order of the iteration:
    # Outer should be iter, inner should be strat, because within each you want to start and end at same point but you want that point to change from iter to iter
    
    sto = 1
    for iS in range(nS):
        print 'Iteration {}'.format(iS)
        Res = {sgy: [] for sgy in sgys}
        Loc = {sgy: [] for sgy in sgys}
        QQQ = {sgy: [] for sgy in sgys}
        
        # Create environment, agent, and simulator
        e = Environment()  # create environment (also adds some dummy traffic)
        a = e.create_agent(LearningAgent)  # create agent
        e.set_primary_agent(a, enforce_deadline=True)  # set agent to track
        sim = Simulator(e, update_delay=0.011)  # reduce update_delay to speed up simulation
        sim.run(n_trials=1)  # press Esc or close pygame window to quit
        
#        global stt0, dst0
#        stt = stt0
#        dst = dst0
        
        stt = e.stt
        dst = e.dst
        print "stt: ", stt
        print "dst: ", dst
        for i, sgy in enumerate(sgys):
            
            # Assign strategy
            strategy = sgy
            
            # Run simulation
#            print "Agent_pst: Start {}, Dest {}".format(e.stt, e.dst)
#            if i==0: # this will start each simulation from a different point
#            if (i==0) & (iS==0): # this will start each simulation from the same different point
#                sim.run(n_trials=1)  # press Esc or close pygame window to quit
#                stt = e.stt
#                dst = e.dst
#            else:
#                sim.run(n_trials=1, stt=stt, dst=dst)  # press Esc or close pygame window to quit
            
            # Use this to impose starting point and destination
            sim.run(n_trials=1, stt=stt, dst=dst)  # press Esc or close pygame window to quit
                
            T_tot.ix[iS, sgy] = e.time_tot
            T_lft.ix[iS, sgy] = e.time_lft
        
            print "Finished simulation, strategy = '%s'" % (strategy)
            
            if sto:
                Res[sgy].append(a.A)
                Loc[sgy].append(e.loc)
                QQQ[sgy].append(a.Q)
            if sgy=='qlearner':
                global Q_est
                Q_est = a.Q[-1]
        
        if sto:
            R4.append(Res)
            L4.append(Loc)
            Q4.append(QQQ)
    
    return (R4, L4, Q4, T_tot, T_lft)
        
    
   
        
if __name__ == '__main__':
    R4, L4, Q4, T_tot, T_lft = run()
    
    # REM
    # iS = 3, iSS = 2
    # Q4[iS]['stubborn'][0][iSS]
    # Q matrix for iteration iS, strategy 'stubborn', sub-iteration (step) iSS = matrix n_states x n_actions
    # L4[iS]['stubborn']: path for iteration iS and strategy subborn
    # R4[iS]['stubborn']: data frame with all inputs, action taken, and reward
    
    plo = 1
    dt0 = '20160608'
    if plo:
        # Would be probably interesting to see the fan of paths if also all the simulations start from the same place
        global stt0, dst0
        fig = plt.figure(figsize=(8,8))
        for (k, sgy) in enumerate(sgys):
            ax = fig.add_subplot(2,2,k+1)
            for iS in range(len(L4)):
                x, y = [], []
                for z in L4[iS][sgy][0]:
                    x.append(z[0])
                    y.append(z[1])
                ax.plot(x, y)
#            ax.plot(stt0[0], stt0[1], 'k', marker='o', ms=10, markerfacecolor=None, label='Start')
#            ax.plot(dst0[0], dst0[1], 'k', marker='s', ms=10, markerfacecolor=None, label='Stop' )
            ax.set_title(sgy)
            ax.set_xticks(range(10))
            ax.set_xticks(range(8))
            ax.set_xlim((0, 10))
            ax.set_ylim((0, 8))
#        plt.savefig(r'C:\Users\rghiglia\Documents\ML_ND\ML_ND_Projects\smartcab\Paths_' + dt0 + '_.png', bbox_inches='tight')
        
#        tot_time = len(R4[iS]['stubborn'][0]) # this is a shortcut: you should record that properly. This is assuming that this strategy always fails. I think that is ok because the code discards destinations that are too close to the starting point; I didn't get the T_tot and T_lft ...
        nS = len(R4)
        succ = pd.DataFrame({sgy: np.zeros((nS,1))[:,0] for sgy in sgys})
        fig = plt.figure(figsize=(8,8))
        for (k, sgy) in enumerate(sgys):
            ax = fig.add_subplot(2,2,k+1)
            for iS in range(len(L4)):
                ax.plot(R4[iS][sgy][0]['cum_rwd'])
            ax.set_title(sgy.title())
            ax.set_xlim((0, 35))
            ax.set_ylim((-20, 50))
            ax.set_ylabel('Cumulative reward')
            if k>=2:
                ax.set_xlabel('Time')
#        plt.savefig(r'C:\Users\rghiglia\Documents\ML_ND\ML_ND_Projects\smartcab\Rewards_' + dt0 + '_.png', bbox_inches='tight')
    
    Eff = T_lft / T_tot
    succ = 1 - (T_lft==0).sum()/len(T_tot)
    print "\nEfficiency: "
    print T_lft / T_tot
    print "\nSuccess rate: "
    print succ
#        succ_rate = succ.sum() / len(succ)
    
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    iS = 9
    
    fig = plt.figure(figsize=(12,8))
    for (k, sgy) in enumerate(sgys):
        ax = fig.add_subplot(2,2,k+1, projection='3d')
        QQ = Q4[iS][sgy][0]     # a list of data frames Q: n_states x n_actions
        lbls = [st + '_' + a for st in QQ[0].index for a in QQ[0].columns]
        xs = np.arange(len(lbls))
    
        nS_tmp = len(QQ)
        for k in range(0, nS_tmp, 5):
            z_tmp = QQ[k].values.flatten()
            ax.bar(xs[z_tmp>0], z_tmp[z_tmp>0], k, zdir='y', color='b', alpha=0.4)
            ax.bar(xs[z_tmp<=0], z_tmp[z_tmp<=0], k, zdir='y', color='r', alpha=0.4)
        plt.xticks(xs, lbls)
        ax.set_title("Evolution of Q: strategy = %s " % sgy)
#    plt.savefig(r'C:\Users\rghiglia\Documents\ML_ND\ML_ND_Projects\smartcab\Q_20160606_{}.png'.format(iS), bbox_inches='tight')
    
    
    