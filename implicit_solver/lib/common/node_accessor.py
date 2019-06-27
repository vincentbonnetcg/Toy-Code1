"""
@author: Vincent Bonnet
@description : This class provides a mapping between node identifiers and data
"""

from lib.system.scene import Scene

def node_id(scene : Scene, object_id, local_node_id):
    global_node_id = scene.dynamics[object_id].node_global_offset + local_node_id
    return [object_id, local_node_id, global_node_id]

def node_global_index(node_id):
    return node_id[2]

def node_state(scene : Scene, node_id):
    dynamic = scene.dynamics[node_id[0]]
    x = dynamic.x[node_id[1]]
    v = dynamic.v[node_id[1]]
    return (x, v)

def node_add_f(scene : Scene, node_id, force):
    dynamic = scene.dynamics[node_id[0]]
    dynamic.f[node_id[1]] += force
