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
    free_index=-1
    if gibbs_neighbors==True and counter==1:
        for index in range(0,len(x)):
            if x[index]!=y[index]:
                free_index=index
            else:
                continue
    return free_index


#I will use two dictonaries to store the essential information 1st: Π[(1,1)]=0.1 for example, storing the state's corresponding stationary probability 
# the second dictionary is a nested dictionary {x1:{x2:0.2,x3:0.4....},...}, similar to a probability matrix
def nested():
    return defaultdict(int)
# This is an implementaton of the Gibbs-Sampling procedure
# The MC has n**dim many states; the target stationary distribution is pi
# The third_variable_is when set to True, prints the various items involved in the procedure
# (not advisable to print for large MCs)
def probability_calculation(state_dict,gibbs_neighbors_array,initial_state,target_state,d):#gibbs neighbors array must contain the initial state itself
    #initial state should not be in gibbs neighbors_array
    #Pxy=1/d*p(y)/p(x2,x3,...xd)=1/d*p(y)/(px1+px2...),where x1 x2... share the same free index with y and intial state, initial state is also included in the denominator
    neighbors_probability=[]
    neighbors_probability.append(state_dict[initial_state])
    neighbors_probability.append(state_dict[target_state])
    #since free_coordinates_gibbs_neighbors only return those different states with same free index with target state and initial state.
    free_index=free_coordinates_of_gibbs_neighbors(initial_state,target_state)
    for each_neighbor in gibbs_neighbors_array:
        if free_coordinates_of_gibbs_neighbors(each_neighbor,target_state)==free_index and free_coordinates_of_gibbs_neighbors(each_neighbor,initial_state)==free_index:
            neighbors_probability.append(state_dict[each_neighbor])
       #     print('the neighbor added to the array is', each_neighbor)
        else:
            continue
    ans=(1/d)*state_dict[target_state]/sum(neighbors_probability)
    #only calculates the neighbors instead the identical case the indentical case will be 1-sum...
    return ans

def create_gibbs_MC(n, dim, pi, do_want_to_print) :
    states=[]
    for x in it.product(range(n), repeat = dim) :
        states.append(x)
    #states is an array storing all the potential states
    state_dict=defaultdict(int)
    probability_dict=defaultdict(nested)
    for i in range(0,len(states)):#i is the index in state array
        state_dict[states[i]]=pi[i]
    #print(state_dict)
    for x in states:
        gibbs_neighbor_array=[]
        for y in states:
            check_result,counter=check_if_these_states_are_gibbs_neighbors(x,y)
            if check_result==True and counter==1:# neglecting itself
                gibbs_neighbor_array.append(y)
            else:
                continue
        for each_neighbor in gibbs_neighbor_array:
            probability_dict[x][each_neighbor]=probability_calculation(state_dict,gibbs_neighbor_array,x,each_neighbor,dim)
        probability_dict[x][x]=1-sum(probability_dict[x].values())
    probability_matrix=[]
    for i in states:
        temp=[]
        for j in states:
            temp.append(probability_dict[i][j])
        probability_matrix.append(temp)   
    if (do_want_to_print) :
        print ("Generating the Probability Matrix using Gibbs-Sampling")
        print ("Target Stationary Distribution:",pi)
        #print('probability_matrix=',np.matrix(probability_matrix))
    return probability_matrix


# Trial 1... States: {(0,0), (0,1), (1,0), (1,1)} (i.e. 4 states)
n = 2
dim = 2
a = generate_a_random_probability_vector(n**dim)
print('this is the n=',n,'dim=',dim,'case')
print("(Random) Target Stationary Distribution\n", a)
p = create_gibbs_MC(n, dim, a, True) 
print ("Probability Matrix:")
print (np.matrix(p))
print ("Does the Probability Matrix have the desired Stationary Distribution?", np.allclose(np.matrix(a), np.matrix(a)* np.matrix(p)))

# Trial 2... States{(0,0), (0,1),.. (0,9), (1,0), (1,1), ... (9.9)} (i.e. 100 states)
n = 10
dim = 2
print('this is the n=',n,'dim=',dim,'case','with no do_want_to_print')
a = generate_a_random_probability_vector(n**dim)
p = create_gibbs_MC(n, dim, a, False) 
print ("Does the Probability Matrix have the desired Stationary Distribution?", np.allclose(np.matrix(a), np.matrix(a)* np.matrix(p)))

# Trial 3... 1000 states 
n = 10
dim = 3
print('this is the n=',n,'dim=',dim,'case')
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
print('this is the n=',n,'dim=',dim,'case')
t1 = time.time()
a = generate_a_random_probability_vector(n**dim)
p = create_gibbs_MC(n, dim, a, False) 
t2 = time.time()
hours, rem = divmod(t2-t1, 3600)
minutes, seconds = divmod(rem, 60)
print ("It took ", hours, "hours, ", minutes, "minutes, ", seconds, "seconds to finish this task")
print ("Does the Probability Matrix have the desired Stationary Distribution?", np.allclose(np.matrix(a), np.matrix(a)* np.matrix(p)))

