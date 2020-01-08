"""
@author: Vincent Bonnet
@description : Bending Constraint for the implicit solver
"""

import numpy as np

from lib.objects.components import ConstraintBase
import lib.common.jit.math_2d as math2D
import lib.common.jit.node_accessor as na
from lib.system.scene import Scene
import lib.objects.components.jit.bending_lib as bending_lib
import lib.common.code_gen as generate

class Bending(ConstraintBase):
    '''
    Describes a 2D bending constraint of a thin inextensible wire
    between three nodes.
    This bending is NOT the proper bending formulation and uses angle instead of curvature
    Some instabilities when using the curvature => Need to investigate
    '''
    def __init__(self):
        '''
        Constraint three nodes to maintain angle between
        node_ids[0] - node_ids[1] - node_ids[2]
        '''
        ConstraintBase.__init__(self, num_nodes = 3)
        self.rest_angle = np.float64(0.0)

    @classmethod
    def pre_compute(cls, scene, details, np_block_ids):
        pass

    @classmethod
    def compute_gradients(cls, details, np_block_ids):
        compute_bending_forces(details.bending, details.node, np_block_ids)

    @classmethod
    def compute_hessians(cls, details, np_block_ids):
        compute_bending_hessians(details.bending, details.node, np_block_ids)

@generate.as_vectorized(njit=True, parallel=False, debug=False, block_ids=True)
def compute_bending_rest(bending : Bending, detail_nodes):
    x0 = na.node_x(detail_nodes, bending.node_IDs[0])
    x1 = na.node_x(detail_nodes, bending.node_IDs[1])
    x2 = na.node_x(detail_nodes, bending.node_IDs[2])
    bending.rest_angle = np.float64(math2D.angle(x0, x1, x2))

@generate.as_vectorized(njit=True, parallel=False, debug=False, block_ids=True)
def compute_bending_forces(bending : Bending, detail_nodes):
    x0 = na.node_x(detail_nodes, bending.node_IDs[0])
    x1 = na.node_x(detail_nodes, bending.node_IDs[1])
    x2 = na.node_x(detail_nodes, bending.node_IDs[2])
    forces = bending_lib.elastic_bending_forces(x0, x1, x2, bending.rest_angle, bending.stiffness, (True, True, True))
    bending.f[0] = forces[0]
    bending.f[1] = forces[1]
    bending.f[2] = forces[2]

@generate.as_vectorized(njit=True, parallel=False, debug=False, block_ids=True)
def compute_bending_hessians(bending : Bending, detail_nodes):
    x0 = na.node_x(detail_nodes, bending.node_IDs[0])
    x1 = na.node_x(detail_nodes, bending.node_IDs[1])
    x2 = na.node_x(detail_nodes, bending.node_IDs[2])
    dfdx = bending_lib.elastic_bending_numerical_hessians(x0, x1, x2, bending.rest_angle, bending.stiffness)
    bending.dfdx[0][0] = dfdx[0]
    bending.dfdx[1][1] = dfdx[1]
    bending.dfdx[2][2] = dfdx[2]
    bending.dfdx[0][1] = bending.dfdx[1][0] = dfdx[3]
    bending.dfdx[0][2] = bending.dfdx[2][0] = dfdx[4]
    bending.dfdx[1][2] = bending.dfdx[2][1] = dfdx[5]

