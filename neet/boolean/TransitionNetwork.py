# TransitionNetwork.py
#
# Bryan Daniels
# 2023/5/15
#
# Moving Sabah Ashfeen's code for creating a logic network
# from a transition list to a more convenient location.
#

from neet import boolean
import numpy as np

# contruct logic table from transition list
def transitions_to_logic_table(transitions,decode):
    """
    Construct a logic table given a list of all transitions.
    
    transitions     : List of length 2^(# of nodes) specifying the
                      resulting encoded state at t+1 given
                      each initial state at time t.
    decode          : A function that produces the network state as
                      a binary list given the network state encoded
                      as an integer between 0 and 2^(# of nodes) - 1.
                      For a Neet network `net`, this is given by
                      `net.decode`.
    """
    result = []
    inner_tuple = [] #each node will hold every other node as input
    if transitions is None:
        return
    states = [decode(x) for x in transitions]
    num_nodes = len(states[0])
    #fill inner_tuple
    for i in range(0,num_nodes):
        inner_tuple.append(i)

    #looping through each node
    for node in range(0,num_nodes):
        outer_tuple = []
        outer_tuple.append(tuple(inner_tuple))
        binary_set = []
        #checking each state for that node
        for input,state in enumerate(states):
            #checking if node is active in that particular state
            if state[node] == 1:
                #add index of that state as a rule, should be in binary
                #this assumes that each node depends on every other node
                binary = decode(input)
                binary_set.append(''.join([str(x) for x in binary]))
        outer_tuple.append(set(binary_set))
        result.append(tuple(outer_tuple))

    return result

def transitions_to_net(transitions):
    """
    Creates a Neet network with the given transition map.
    
    transitions     : List of length 2^(# of nodes) specifying the
                      resulting encoded state at t+1 given
                      each initial state at time t.
    """
    # determine the number of nodes in the network
    num_nodes = np.log2(len(transitions))
    assert(num_nodes == int(num_nodes))
    num_nodes = int(num_nodes)
    
    # a somewhat silly way to get consistent decoding:
    # define a dummy neet network of the correct size
    decode = boolean.WTNetwork(
                np.zeros((num_nodes,num_nodes))).decode
    
    logic_table = transitions_to_logic_table(transitions,decode)

    return boolean.LogicNetwork(logic_table)
