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
    def pre_compute(cls, blocks_iterator, scene, details) -> None:
        pass

    @classmethod
    def compute_forces(cls, blocks_iterator, details) -> None:
        '''
        Add the force to the datablock
        '''
        for ct_block in blocks_iterator:
            node_ids_ptr = ct_block['node_IDs']
            rest_angle_ptr = ct_block['rest_angle']
            stiffness_ptr = ct_block['stiffness']
            force_ptr = ct_block['f']
            block_n_elements = ct_block['blockInfo_numElements']

            for ct_index in range(block_n_elements):
                x0, v0 = na.node_xv(details.node.blocks, node_ids_ptr[ct_index][0])
                x1, v1 = na.node_xv(details.node.blocks, node_ids_ptr[ct_index][1])
                x2, v2 = na.node_xv(details.node.blocks, node_ids_ptr[ct_index][2])
                f0, f1, f2 = bending_lib.elastic_bending_forces(x0, x1, x2, rest_angle_ptr[ct_index], stiffness_ptr[ct_index], (True, True, True))
                force_ptr[ct_index][0] = f0
                force_ptr[ct_index][1] = f1
                force_ptr[ct_index][2] = f2

    @classmethod
    def compute_jacobians(cls, blocks_iterator, details) -> None:
        '''
        Add the force jacobian functions to the datablock
        '''
        for ct_block in blocks_iterator:
            node_ids_ptr = ct_block['node_IDs']
            rest_angle_ptr = ct_block['rest_angle']
            stiffness_ptr = ct_block['stiffness']
            dfdx_ptr = ct_block['dfdx']
            block_n_elements = ct_block['blockInfo_numElements']

            for ct_index in range(block_n_elements):
                x0, v0 = na.node_xv(details.node.blocks, node_ids_ptr[ct_index][0])
                x1, v1 = na.node_xv(details.node.blocks, node_ids_ptr[ct_index][1])
                x2, v2 = na.node_xv(details.node.blocks, node_ids_ptr[ct_index][2])
                dfdx = bending_lib.elastic_bending_numerical_jacobians(x0, x1, x2, rest_angle_ptr[ct_index], stiffness_ptr[ct_index])
                dfdx_ptr[ct_index][0][0] = dfdx[0]
                dfdx_ptr[ct_index][1][1] = dfdx[1]
                dfdx_ptr[ct_index][2][2] = dfdx[2]
                dfdx_ptr[ct_index][0][1] = dfdx_ptr[ct_index][1][0] = dfdx[3]
                dfdx_ptr[ct_index][0][2] = dfdx_ptr[ct_index][2][0] = dfdx[4]
                dfdx_ptr[ct_index][1][2] = dfdx_ptr[ct_index][2][1] = dfdx[5]

@generate.as_vectorized(njit=True, parallel=False, debug=False, block_ids=True)
def compute_bending_rest(bending : Bending, detail_nodes):
    x0 = na.node_x(detail_nodes, bending.node_IDs[0])
    x1 = na.node_x(detail_nodes, bending.node_IDs[1])
    x2 = na.node_x(detail_nodes, bending.node_IDs[2])
    bending.rest_angle = np.float64(math2D.angle(x0, x1, x2))
