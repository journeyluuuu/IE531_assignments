import sys
import argparse
import random
import numpy as np 
import time
sys.setrecursionlimit(3000)


#correct result
def randomized_select(current_array, k) :
    if (len(current_array) == 1) :
        return current_array[0]
    else : 
        # pick a random pivot-element
        p = current_array[random.randint(0,len(current_array)-1)]

        # split the current_array into three sub-arrays: Less_than_p, Equal_to_p and Greater_than_p
        Less_than_p = []
        Equal_to_p = []
        Greater_than_p = []
        for x in current_array : 
            if (x < p) : 
                Less_than_p.extend([x])
            if (x == p) : 
                Equal_to_p.extend([x])
            if (x > p) : 
                Greater_than_p.extend([x])

        if (k < len(Less_than_p)) :
            return randomized_select(Less_than_p, k)
        elif (k >= len(Less_than_p) + len(Equal_to_p)) : 
            return randomized_select(Greater_than_p, k - len(Less_than_p) - len(Equal_to_p))
        else :
            return p

def sort_and_select(current_array, k) :
    # sort the array
    sorted_current_array = np.sort(current_array)
    return sorted_current_array[k]

number_of_trials = 1000
mean_running_time = []
std_dev_running_time = []
for j in range(1, 40) :
    array_size = 100*j
    k = random.randint(1,array_size)
    # fill the array with random values
    my_array = [random.randint(1,100*array_size) for _ in range(array_size)]
    
    # run a bunch of random trials and get the algorithm's running time
    running_time = []
    for i in range(1, number_of_trials) :
        t1 = time.time()
        answer1 = randomized_select(my_array,k)
        t2 = time.time()
        answer2 = sort_and_select(my_array, k)
        running_time.extend([t2-t1])
        if (answer1 != answer2) :
                print ("Something went wrong")
                exit()
    mean_running_time.extend([np.mean(running_time)])
    std_dev_running_time.extend([np.std(running_time)])
