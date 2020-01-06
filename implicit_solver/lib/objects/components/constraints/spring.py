"""
@author: Vincent Bonnet
@description : Constraint base for the implicit solver
"""

import numpy as np
from lib.objects.components import ConstraintBase
from lib.common.convex_hull import ConvexHull
import lib.common.jit.math_2d as math2D
import lib.common.jit.node_accessor as na
from lib.system import Scene
import lib.objects.components.jit.spring_lib as spring_lib
import lib.common.code_gen as generate


class AnchorSpring(ConstraintBase):
    '''
    Describes a 2D spring constraint between a node and point
    '''
    def __init__(self):
        ConstraintBase.__init__(self, num_nodes = 1)
        self.rest_length = np.float64(0.0)
        self.kinematic_index = np.uint32(0)
        self.kinematic_component_index =  np.uint32(0)
        self.kinematic_component_param = np.float64(0.0)
        self.kinematic_component_pos = np.zeros(2, dtype = np.float64)

    @classmethod
    def pre_compute(cls, blocks_iterator, scene, details) -> None:
        for ct_block in blocks_iterator:
            k_index_ptr = ct_block['kinematic_index']
            k_c_index_ptr = ct_block['kinematic_component_index']
            k_c_param_ptr = ct_block['kinematic_component_param']
            k_c_pos_ptr = ct_block['kinematic_component_pos']
            block_n_elements = ct_block['blockInfo_numElements']

            for ct_index in range(block_n_elements):
                kinematic = scene.kinematics[k_index_ptr[ct_index]]
                point_params = ConvexHull.ParametricPoint(k_c_index_ptr[ct_index], k_c_param_ptr[ct_index])
                k_c_pos_ptr[ct_index] = kinematic.get_position_from_parametric_point(point_params)

    @classmethod
    def compute_forces(cls, blocks_iterator, details) -> None:
        for ct_block in blocks_iterator:
            kinematic_vel = np.zeros(2)
            node_ids_ptr = ct_block['node_IDs']
            stiffness_ptr = ct_block['stiffness']
            damping_ptr = ct_block['damping']
            rest_length_ptr = ct_block['rest_length']
            k_c_pos_ptr = ct_block['kinematic_component_pos']
            force_ptr = ct_block['f']
            block_n_elements = ct_block['blockInfo_numElements']

            for ct_index in range(block_n_elements):
                node_ids = node_ids_ptr[ct_index]
                x, v = na.node_xv(details.node.blocks, node_ids[0])
                target_pos = k_c_pos_ptr[ct_index]
                force = spring_lib.spring_stretch_force(x, target_pos, rest_length_ptr[ct_index], stiffness_ptr[ct_index])
                force += spring_lib.spring_damping_force(x, target_pos, v, kinematic_vel, damping_ptr[ct_index])
                force_ptr[ct_index] = force

    @classmethod
    def compute_jacobians(cls, blocks_iterator, details) -> None:
        for ct_block in blocks_iterator:
            kinematic_vel = np.zeros(2)
            node_ids_ptr = ct_block['node_IDs']
            stiffness_ptr = ct_block['stiffness']
            damping_ptr = ct_block['damping']
            rest_length_ptr = ct_block['rest_length']
            k_c_pos_ptr = ct_block['kinematic_component_pos']
            dfdx_ptr = ct_block['dfdx']
            dfdv_ptr = ct_block['dfdv']
            block_n_elements = ct_block['blockInfo_numElements']

            for ct_index in range(block_n_elements):
                node_ids = node_ids_ptr[ct_index]
                x, v = na.node_xv(details.node.blocks, node_ids[0])
                target_pos = k_c_pos_ptr[ct_index]
                dfdx = spring_lib.spring_stretch_jacobian(x, target_pos, rest_length_ptr[ct_index], stiffness_ptr[ct_index])
                dfdv = spring_lib.spring_damping_jacobian(x, target_pos, v, kinematic_vel, damping_ptr[ct_index])
                dfdx_ptr[ct_index][0][0] = dfdx
                dfdv_ptr[ct_index][0][0] = dfdv

class Spring(ConstraintBase):
    '''
    Describes a 2D spring constraint between two nodes
    '''
    def __init__(self):
        ConstraintBase.__init__(self, num_nodes = 2)
        self.rest_length = np.float64(0.0)

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
            rest_length_ptr = ct_block['rest_length']
            stiffness_ptr = ct_block['stiffness']
            damping_ptr = ct_block['damping']
            force_ptr = ct_block['f']
            block_n_elements = ct_block['blockInfo_numElements']

            for ct_index in range(block_n_elements):
                node_ids = node_ids_ptr[ct_index]
                x0, v0 = na.node_xv(details.node.blocks, node_ids[0])
                x1, v1 = na.node_xv(details.node.blocks, node_ids[1])
                force = spring_lib.spring_stretch_force(x0, x1, rest_length_ptr[ct_index], stiffness_ptr[ct_index])
                force += spring_lib.spring_damping_force(x0, x1, v0, v1, damping_ptr[ct_index])
                force_ptr[ct_index][0] = force
                force_ptr[ct_index][1] = force * -1.0

    @classmethod
    def compute_jacobians(cls, blocks_iterator, details) -> None:
        '''
        Add the force jacobian functions to the datablock
        '''
        for ct_block in blocks_iterator:
            node_ids_ptr = ct_block['node_IDs']
            rest_length_ptr = ct_block['rest_length']
            stiffness_ptr = ct_block['stiffness']
            damping_ptr = ct_block['damping']
            dfdx_ptr = ct_block['dfdx']
            dfdv_ptr = ct_block['dfdv']
            block_n_elements = ct_block['blockInfo_numElements']

            for ct_index in range(block_n_elements):
                x0, v0 = na.node_xv(details.node.blocks, node_ids_ptr[ct_index][0])
                x1, v1 = na.node_xv(details.node.blocks, node_ids_ptr[ct_index][1])
                dfdx = spring_lib.spring_stretch_jacobian(x0, x1, rest_length_ptr[ct_index], stiffness_ptr[ct_index])
                dfdv = spring_lib.spring_damping_jacobian(x0, x1, v0, v1, damping_ptr[ct_index])
                # Set jacobians
                dfdx_ptr[ct_index][0][0] = dfdx_ptr[ct_index][1][1] = dfdx
                dfdx_ptr[ct_index][0][1] = dfdx_ptr[ct_index][1][0] = dfdx * -1
                dfdv_ptr[ct_index][0][0] = dfdv_ptr[ct_index][1][1] = dfdv
                dfdv_ptr[ct_index][0][1] = dfdv_ptr[ct_index][1][0] = dfdv * -1

@generate.as_vectorized(njit=True, parallel=False, debug=False, block_ids=True)
def compute_anchor_spring_rest(anchor_spring : AnchorSpring, detail_nodes):
    x = na.node_x(detail_nodes, anchor_spring.node_IDs[0])
    anchor_spring.rest_length = np.float64(math2D.distance(anchor_spring.kinematic_component_pos, x))

@generate.as_vectorized(njit=True, parallel=False, debug=False, block_ids=True)
def compute_spring_rest(spring : Spring, detail_nodes):
    x0 = na.node_x(detail_nodes, spring.node_IDs[0])
    x1 = na.node_x(detail_nodes, spring.node_IDs[1])
    spring.rest_length = math2D.distance(x0, x1)

