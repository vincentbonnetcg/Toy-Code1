"""
@author: Vincent Bonnet
@description : Constraint base for the implicit solver
"""

from lib.objects.components import ConstraintBase
import lib.common.jit.math_2d as math2D
import lib.common.jit.node_accessor as na
from lib.system.scene import Scene
import numpy as np
import lib.objects.components.jit.area_lib as area_lib
import lib.common.code_gen as generate

class Area(ConstraintBase):
    '''
    Describes a 2D area constraint between three nodes
    '''
    def __init__(self):
        ConstraintBase.__init__(self, num_nodes = 3)
        self.rest_area = np.float64(0.0)

    @classmethod
    def pre_compute(cls, blocks_iterator, scene, details, block_ids=None) -> None:
        pass

    @classmethod
    def compute_forces(cls, blocks_iterator, details, block_ids=None) -> None:
        '''
        Add the force to the datablock
        '''
        for ct_block in blocks_iterator:
            node_ids_ptr = ct_block['node_IDs']
            rest_area_ptr = ct_block['rest_area']
            stiffness_ptr = ct_block['stiffness']
            force_ptr = ct_block['f']
            block_n_elements = ct_block['blockInfo_numElements']

            for ct_index in range(block_n_elements):
                x0, v0 = na.node_xv(details.node.blocks, node_ids_ptr[ct_index][0])
                x1, v1 = na.node_xv(details.node.blocks, node_ids_ptr[ct_index][1])
                x2, v2 = na.node_xv(details.node.blocks, node_ids_ptr[ct_index][2])
                f0, f1, f2 = area_lib.elastic_area_forces(x0, x1, x2, rest_area_ptr[ct_index], stiffness_ptr[ct_index], (True, True, True))
                force_ptr[ct_index][0] = f0
                force_ptr[ct_index][1] = f1
                force_ptr[ct_index][2] = f2

    @classmethod
    def compute_jacobians(cls, blocks_iterator, details, block_ids=None) -> None:
        '''
        Add the force jacobian functions to the datablock
        '''
        for ct_block in blocks_iterator:
            node_ids_ptr = ct_block['node_IDs']
            rest_area_ptr = ct_block['rest_area']
            stiffness_ptr = ct_block['stiffness']
            dfdx_ptr = ct_block['dfdx']
            block_n_elements = ct_block['blockInfo_numElements']

            for ct_index in range(block_n_elements):
                x0, v0 = na.node_xv(details.node.blocks, node_ids_ptr[ct_index][0])
                x1, v1 = na.node_xv(details.node.blocks, node_ids_ptr[ct_index][1])
                x2, v2 = na.node_xv(details.node.blocks, node_ids_ptr[ct_index][2])
                dfdx = area_lib.elastic_area_numerical_jacobians(x0, x1, x2, rest_area_ptr[ct_index], stiffness_ptr[ct_index])
                dfdx_ptr[ct_index][0][0] = dfdx[0]
                dfdx_ptr[ct_index][1][1] = dfdx[1]
                dfdx_ptr[ct_index][2][2] = dfdx[2]
                dfdx_ptr[ct_index][0][1] = dfdx_ptr[ct_index][1][0] = dfdx[3]
                dfdx_ptr[ct_index][0][2] = dfdx_ptr[ct_index][2][0] = dfdx[4]
                dfdx_ptr[ct_index][1][2] = dfdx_ptr[ct_index][2][1] = dfdx[5]

@generate.as_vectorized(njit=True, parallel=False, debug=False, block_ids=True)
def compute_area_rest(area : Area, detail_nodes):
    x0 = na.node_x(detail_nodes, area.node_IDs[0])
    x1 = na.node_x(detail_nodes, area.node_IDs[1])
    x2 = na.node_x(detail_nodes, area.node_IDs[2])
    area.rest_area = np.float64(math2D.area(x0, x1, x2))

