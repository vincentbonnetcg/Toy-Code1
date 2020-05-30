"""
@author: Vincent Bonnet
@description : Graph optimization (Greedy colouring algorithm)
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

'''
 Global Parameters
'''
NUM_NODES = 100
NODE_SEARCH_RADIUS = 0.2

def get_graph():
    return nx.random_geometric_graph(NUM_NODES, NODE_SEARCH_RADIUS, seed=5)

def compute_groups(graph):
    '''
    greedy colouring algorithm
    '''
    nx.set_node_attributes(graph, -1, "group")
    group_ids = nx.get_node_attributes(graph, "group")

    for node, adjacencies in graph.adjacency():
        # get group ids from adjacencies
        adjacency_groups = []
        for adj in adjacencies:
            adjacency_groups.append(group_ids[adj])

        # search for unassigned group id
        group_id = 0
        while group_id in adjacency_groups:
            group_id += 1

        group_ids[node] = group_id

    return group_ids

def show(graph):
    num_nodes = graph.number_of_nodes()
    colours = np.zeros(graph.number_of_nodes())
    # colour from groups
    group_ids = compute_groups(graph)
    #group_ids = nx.coloring.greedy_color(graph) # to compare with own implementation
    for i in range(num_nodes):
        colours[i] = group_ids[i]

    # display the graph
    # Only support up to 20 difference colours (see cmap=plt.cm.tab20)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.axis('equal')

    pos = nx.get_node_attributes(graph, "pos")
    nx.draw_networkx_nodes(graph, pos, node_color=colours, cmap=plt.cm.tab20, node_size = 20, ax=ax)
    nx.draw_networkx_edges(graph, pos, alpha=0.4, ax=ax)

    font = {'family': 'serif',
            'color':  'darkblue',
            'weight': 'normal',
            'size': 14 }
    num_colours = np.max(list(group_ids.values()))
    plt.title(('Greedy Coloring Algorithm (%d colours)'%num_colours), fontdict=font)
    # plt.legend(bbox_to_anchor=(1, 1), loc=2)
    plt.axis('off')


if __name__ == '__main__':
    graph = get_graph()
    show(graph)

