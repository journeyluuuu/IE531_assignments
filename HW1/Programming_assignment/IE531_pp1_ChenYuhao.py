#!/usr/bin/env python
# coding: utf-8

import argparse
import sys
import math
from collections import deque
import random
# See https://www.geeksforgeeks.org/deque-in-python/ for details on Deques




def initialize(n):# n is pan number, K is deque/peq number both are inputted with input function, need to change to arg
    for i in range(K):
        X=deque()
        if i==0:
            for i in range(n):
                X.append(i+1)
        Towers.append(X)
    return Towers



def is_everything_legal() :
    result = True
    for i in range(K) :#K is the pegs number
        for j in range(len(Towers[i])) :
            for j2 in range(j,len(Towers[i])) :
                if (Towers[i][j2] < Towers[i][j]) :
                    result = False
    return(result)
def move_top_disk(source, dest):
    global number_of_steps 
    number_of_steps = number_of_steps + 1
    x = Towers[source].popleft()
    Towers[dest].appendleft(x)
    if (True == is_everything_legal()) :
        y = " (Legal)"
    else :
        y = " (Illegal)"
    print ("Move disk " + str(x) + " from Peg " + str(source+1) + " to Peg " + str(dest+1) + y)
def print_peg_state(m) :
    global number_of_steps
    print ("-----------------------------")
    print ("State of Peg " + str(m+1) + " (Top to Bottom): " + str(Towers[m]))
    print ("Number of Steps = " + str(number_of_steps))
    print ("-----------------------------")




def move_using_three_pegs(number_of_disks, source, dest, intermediate) :
    if (1 == number_of_disks) :
        move_top_disk (source, dest)
    else :
        move_using_three_pegs (number_of_disks-1, source, intermediate, dest);
        move_top_disk(source, dest)
        move_using_three_pegs (number_of_disks-1, intermediate, dest, source)
def move_using_four_pegs(number_of_disks, source, dest, intermediate1, intermediate2) :
    if (number_of_disks > 0) :
        k = math.floor(math.sqrt(2*number_of_disks))
        move_using_four_pegs(number_of_disks-k, source, intermediate1, intermediate2, dest)
        move_using_three_pegs(k, source, dest, intermediate2)
        move_using_four_pegs(number_of_disks-k, intermediate1, dest, intermediate2, source)
def move_using_more_pegs(number_of_disks,source,dest,intermediates):
# intermediates already get rid of source dest in the first input checked on 1/28 : 0 1 2 3 peq for k==4, cut to 1 2 
    #while number_of_disks >0 :
    if number_of_disks==1:
        move_top_disk(source,dest)
    else:
        if len(intermediates)>1:
            p=math.floor(number_of_disks/2)
        else:
            p=number_of_disks-1
        intermediates_step1=intermediates.copy()# intermediates already get rid of source dest in the first input
        middle=intermediates_step1.pop()
        move_using_more_pegs(p,source,middle,intermediates_step1)
        #step2 
        intermediates_step2=intermediates.copy()
        intermediates_step2.remove(middle)   
        move_using_more_pegs(number_of_disks-p,source,dest,intermediates_step2)
    #step 3, from middle to destination intermediates should get rid of middle, dest but adding the initial source
        intermediates_step3=intermediates.copy()
        intermediates_step3.remove(middle)
        intermediates_step3.appendleft(source)
        move_using_more_pegs(p,middle,dest,intermediates_step3)


def move_using_general_pegs(number_of_disks,source,dest):
    pegs=deque()
    for i in range(dest+1):#dest=3 for 4 pegs case 0, 1,2,3
        pegs.append(i)
    #pegs: 0...dest, dest=3, 0,1,2,3 for the 4 pegs case
    intermediates=pegs.copy()
    intermediates.remove(source)
    intermediates.remove(dest)
    if len(intermediates)>2:
        move_using_more_pegs(number_of_disks,source,dest,intermediates)
    if len(intermediates)==2:
        move_using_four_pegs(number_of_disks,source,dest,intermediates[0],intermediates[1])
    if len(intermediates)==1:
        move_using_three_pegs(number_of_disks,source,dest,intermediates[0])





# Doing the needful to move 5-many disks from the leftmost-peg to the rightmost-peg, using legal-moves for the 4-Peg Tower of Hanoi Problem... 
Towers = deque()

# Global variable that keeps track of the number of steps in our solution 
number_of_steps = 0

# It is always a good idea to set a limit on the depth-of-the-recursion-tree in Python
sys.setrecursionlimit(30000)

argparser=argparse.ArgumentParser()
argparser.add_argument("integer",metavar='N',type=int,nargs='+',help='The first input is the number of disks on the source peg, the second input is the peque number, both input should be integer')
args=argparser.parse_args()
n=args.integer[0]
K=args.integer[1]


Towers=initialize(n)
dest=K-1 # for python notation, K peques, 0...K-1
source=0
number_of_disks=n



print_peg_state(source)
move_using_general_pegs(number_of_disks,source,dest)
print_peg_state(dest)

