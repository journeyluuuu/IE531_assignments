#!/usr/bin/env python3
# coding: utf-8

# Gibbs-Sampling procedure to compute the Probability Matrix of a Discrete-Time Markov
# Chain whose states are the d-dimensional cartesian product of the form 
# {0,1,...n-1} x {0,1,...n-1} x ... X {0,1,...n-1} (i.e. d-many products)
# 
# The target stationary distribution is expressed over the n**d many states 
#
# Written by Prof. R.S. Sreenivas for
# IE531: Algorithms for Data Analytics
#

import sys
import argparse
import random
import numpy as np 
import time
import math
import matplotlib.pyplot as plt
import itertools as it
from collections import defaultdict
# need this to keep the matrix print statements to 4 decimal places
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

# This function computes a random n-dimensional probability vector (whose entries sum to 1)
def generate_a_random_probability_vector(n) :
    a=np.random.random(n,)
    y=a/sum(a)
    return y

# Two d-tuples x and y are Gibbs-Neighbors if they are identical, or they differ in value at just
# one coordinate
def check_if_these_states_are_gibbs_neighbors(x, y) :
    # x and y are dim-tuples -- we will assume they are not equal
    # count the number of coordinates where they differ
    counter=0
    for index in range(0,len(x)):
        if x[index]!=y[index]:
            counter+=1
    if counter<=1:
        return True,counter#True on being Gibbs-Neighbors:no bigger than 1 different index
    else:
        return False,counter#False on being gibbs neighbors

# Given two Gibbs-Neighbors -- that are not identical -- find the coordinate where they differ in value
# this is the "free-coordinate" for this pair
def free_coordinates_of_gibbs_neighbors(x, y) :
    # we assume x and y are gibbs neighbors, then the must agree on at least (dim-1)-many coordinates
    # or, they will disagree on just one of the (dim)-many coordinates... we have to figure out which 
    # coordinate/axes is free
    gibbs_neighbors,counter=check_if_these_states_are_gibbs_neighbors(x,y)
    if gibbs_neighbors==False:
        print('they are not gibbs neighbors')
    if gibbs_neighbors==True and counter==1:
        print('they are gibbs neighbors')
        print('x y has equal length is',len(x)==len(y))
        for index in range(0,len(x)):
            if x[index]!=y[index]:
                free_index=index
            else:
                continue
    if gibbs_neighbors==True and counter==0:
        free_index=-1#standing for identical gibbs neighbors
    return free_index


#I will use two dictonaries to store the essential information 1st: Π[(1,1)]=0.1 for example, storing the state's corresponding stationary probability 
# the second dictionary is a nested dictionary {x1:{x2:0.2,x3:0.4....},...}, similar to a probability matrix
def nested():
    return defaultdict(int)
# This is an implementaton of the Gibbs-Sampling procedure
# The MC has n**dim many states; the target stationary distribution is pi
# The third_variable_is when set to True, prints the various items involved in the procedure
# (not advisable to print for large MCs)
def probability_calculation(state_dict,gibbs_neighbors_array,initial_state,target_state):#gibbs neighbors array must contain the initial state itself
    neighbors_probability=[]
    for each_neighbor in gibbs_neighbors_array:
        neighbors_probability.append(state_dict[each_neighbor])
    print('gibbs_neighbors_array',gibbs_neighbors_array,'initial_state is in gibbs_neighbor_array is', initial_state in gibbs_neighbors_array,'target state in gibbs_neigbor_array is',target_state in gibbs_neighbors_array)
    d=len(gibbs_neighbors_array)
    ans=(1/d)*state_dict[target_state]/sum(neighbors_probability)
    return ans
def create_gibbs_MC(n, dim, pi, do_want_to_print) :
    states=[]
    for x in it.product(range(n), repeat = dim) :
        states.append(x)
    #states is an array storing all the potential states
    states=np.array(states)
    state_dict=defaultdict(int)
    probability_dict=defaultdict(nested)
    for i in range(0,len(states)):#i is the index in state array
        state_dict[states[i]]=pi[i]
    for x in states:
        gibbs_neighbor_array=[]
        for y in states:
            check_result,counter=check_if_these_states_are_gibbs_neighbors(x,y)
            if check_result==True:
                gibbs_neighbor_array.append(y)
            else:
                continue
        print(gibbs_neighbor_array,'should be of length d+1',len(gibbs_neighbor_array),dim+1)
        gibbs_neighbor_array=np.array(gibbs_neighbor_array)
        for each_neighbor in gibbs_neighbor_array:
            probability_dict[x][each_neighbor]=probability_calculation(state_dict,gibbs_neighbor_array,x,each_neighbor)
    if (do_want_to_print) :
        print ("Generating the Probability Matrix using Gibbs-Sampling")
        print ("Target Stationary Distribution:",pi)
        print('probability_dict=',probability_dict)
    return probability_dict
        
    # the probability matrix will be (n**dim) x (n**dim) 
    probability_matrix = [[0 for x in range(n**dim)] for y in range(n**dim)]
    
    # the state of the MC is a dim-tuple (i.e. if dim = 2, it is a 2-tuple; if dim = 4, it is a 4-tuple)
    # got this from https://stackoverflow.com/questions/7186518/function-with-varying-number-of-for-loops-python
    for x in it.product(range(n), repeat = dim) :
        # x is a dim-tuple where each variable ranges over 0,1,...,n-1
        for y in it.product(range(n), repeat = dim) :
            ** WRITE THIS PART ***
            ** KEEP IN MIND THAT THE EXAMPLE IN FIGURE 4.3 OF THE BOOK HAS SEVERAL TYPOS**

    return probability_matrix

# Trial 1... States: {(0,0), (0,1), (1,0), (1,1)} (i.e. 4 states)
n = 2
dim = 2
a = generate_a_random_probability_vector(n**dim)
print("(Random) Target Stationary Distribution\n", a)
p = create_gibbs_MC(n, dim, a, True) 
print ("Probability Matrix:")
print (np.matrix(p))
print ("Does the Probability Matrix have the desired Stationary Distribution?", np.allclose(np.matrix(a), np.matrix(a)* np.matrix(p)))

# Trial 2... States{(0,0), (0,1),.. (0,9), (1,0), (1,1), ... (9.9)} (i.e. 100 states)
n = 10
dim = 2
a = generate_a_random_probability_vector(n**dim)
p = create_gibbs_MC(n, dim, a, False) 
print ("Does the Probability Matrix have the desired Stationary Distribution?", np.allclose(np.matrix(a), np.matrix(a)* np.matrix(p)))

# Trial 3... 1000 states 
n = 10
dim = 3
t1 = time.time()
a = generate_a_random_probability_vector(n**dim)
p = create_gibbs_MC(n, dim, a, False) 
t2 = time.time()
hours, rem = divmod(t2-t1, 3600)
minutes, seconds = divmod(rem, 60)
print ("It took ", hours, "hours, ", minutes, "minutes, ", seconds, "seconds to finish this task")
print ("Does the Probability Matrix have the desired Stationary Distribution?", np.allclose(np.matrix(a), np.matrix(a)* np.matrix(p)))

# Trial 4... 10000 states 
n = 10
dim = 4
t1 = time.time()
a = generate_a_random_probability_vector(n**dim)
p = create_gibbs_MC(n, dim, a, False) 
t2 = time.time()
hours, rem = divmod(t2-t1, 3600)
minutes, seconds = divmod(rem, 60)
print ("It took ", hours, "hours, ", minutes, "minutes, ", seconds, "seconds to finish this task")
print ("Does the Probability Matrix have the desired Stationary Distribution?", np.allclose(np.matrix(a), np.matrix(a)* np.matrix(p)))

