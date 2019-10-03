"""
@author: Vincent Bonnet
@description : Implementation of Greedy Colouring Algorithm
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

'''
 Global Parameters
'''
NUM_NODES = 50
NODE_SEARCH_RADIUS = 0.2

'''
 Setup Graph
'''
G = nx.random_geometric_graph(NUM_NODES, NODE_SEARCH_RADIUS)
nx.set_node_attributes(G, "group", -1)

'''
 Greedy Colouring Algorithm
'''
groupIds = nx.get_node_attributes(G, "group")
neighboursList = G.adjacency_list()
for nodeIndex in G.nodes_iter():
    # get group ids from neighbours
    neighbourGroups = []
    for neighbour in neighboursList[nodeIndex]:
        neighbourGroups.append(groupIds[neighbour])
    # search for a group id which is not in the neighbourGroups
    groupId = 0
    while groupId in neighbourGroups:
        groupId += 1
    
    groupIds[nodeIndex] = groupId

'''
 Display
'''
# copy group ids dicionnary to colour ids array
colours = np.zeros(G.number_of_nodes())
for nodeIndex, data in G.nodes_iter(data=True):
    colours[nodeIndex] = groupIds[nodeIndex]

# Display the graph
# Only support up to 20 difference colours (see cmap=plt.cm.tab20)
fig, ax = plt.subplots(figsize=(6,6))
ax.axis('equal')

pos = nx.get_node_attributes(G, "pos")
nx.draw_networkx_nodes(G, pos, node_color=colours, cmap=plt.cm.tab20, node_size = 100, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax)

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16 }
plt.title('Greedy Coloring Algorithm', fontdict=font)
# plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.axis('off')

