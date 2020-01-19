"""
Template for implementing QLearner  (c) 2015 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Tucker Balch (replace with your name)
GT User ID: tb34 (replace with your User ID)
GT ID: 900897987 (replace with your GT ID)
"""

import numpy as np
import random as rand


class QLearner(object):
    # author: Shantanu Singh
    def author(self):
        return "ssingh341"

    def __init__(self,
                 num_states=100,
                 num_actions=4,
                 alpha=0.2,
                 gamma=0.9,
                 rar=0.5,
                 radr=0.99,
                 dyna=0,
                 verbose=False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        if dyna > 0:
            self.Tc = np.ones((num_states, num_actions, num_states))*0.00001
            self.T = np.ones((num_states, num_actions, num_states))
            self.R = np.zeros((num_states, num_actions))
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((num_states, num_actions))

    def querysetstate(self, s):
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Update the state without updating the Q-table  		   	  			  	 		  		  		    	 		 		   		 		  
        @param s: The new state  		   	  			  	 		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			  	 		  		  		    	 		 		   		 		  
        """
        self.s = s
        r_number = rand.random()
        if r_number >= self.rar:
            action = np.argmax(self.Q[s])
        else:
            action = rand.randint(0, self.num_actions-1)
        if self.verbose:
            print(f"s = {s}, a = {action}")
        return action

    def query(self, s_prime, r):
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Update the Q table and return an action  		   	  			  	 		  		  		    	 		 		   		 		  
        @param s_prime: The new state  		   	  			  	 		  		  		    	 		 		   		 		  
        @param r: The ne state  		   	  			  	 		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			  	 		  		  		    	 		 		   		 		  
        """
        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + self.alpha * (
            r + self.gamma * np.max(self.Q[s_prime]))
        if (self.dyna > 0):
            self.Tc[self.s, self.a, s_prime] += 1
            self.T[self.s, self.a, :] = self.Tc[self.s, self.a, :] / \
                (self.Tc[self.s, self.a, :]).sum()
            dyna_alpha = 0.95
            self.R[self.s, self.a] = (
                1 - dyna_alpha) * self.R[self.s, self.a] + r * dyna_alpha
            for i in range(self.dyna):
                rand_s = rand.randint(0, self.num_states - 1)
                rand_a = rand.randint(0, self.num_actions - 1)
                # cum_sum = 0.0
                rand_no = rand.random()
                # for loopsprime in range(self.T.shape[2]):
                #     cum_sum += self.T[rand_s, rand_a, loopsprime]
                #     if cum_sum >= rnd_no:
                #         dyna_s_prime = loopsprime
                #         break
                dyna_s_prime = np.where(
                    np.cumsum(self.T[rand_s, rand_a]) >= rand_no)[0][0]
                dyna_r = self.R[rand_s, rand_a]
                self.Q[rand_s, rand_a] = (1 - self.alpha) * self.Q[rand_s, rand_a] + \
                    self.alpha * (dyna_r + self.gamma *
                                  np.max(self.Q[dyna_s_prime]))
        self.rar *= self.radr
        self.s = s_prime
        r_number = rand.random()
        if r_number >= self.rar:
            # action = np.max(self.Q[s])
            action = self.querysetstate(self.s)
        else:
            action = rand.randint(0, self.num_actions-1)
        self.a = action
        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")
        return action


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
