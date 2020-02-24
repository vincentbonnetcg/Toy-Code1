"""
@author: Vincent Bonnet
@description : Constraint base for the implicit solver
"""

import numpy as np

import lib.common.jit.math_2d as math2D
import lib.common.jit.node_accessor as na
import lib.common.code_gen as generate
import lib.objects.jit.utils.area_lib as area_lib
import lib.objects.jit as cpn

class Area(cpn.ConstraintBase):
    '''
    Describes a 2D area constraint between three nodes
    '''
    def __init__(self):
        cpn.ConstraintBase.__init__(self, num_nodes = 3)
        self.rest_area = np.float64(0.0)

    @classmethod
    def pre_compute(cls):
        return None

    @classmethod
    def compute_rest(cls):
        return compute_area_rest

    @classmethod
    def compute_gradients(cls):
        return compute_area_forces

    @classmethod
    def compute_hessians(cls):
        return compute_area_jacobians

@generate.as_vectorized(block_handles=True)
def compute_area_rest(area : Area, detail_nodes):
    x0 = na.node_x(detail_nodes, area.node_IDs[0])
    x1 = na.node_x(detail_nodes, area.node_IDs[1])
    x2 = na.node_x(detail_nodes, area.node_IDs[2])
    area.rest_area = np.float64(math2D.area(x0, x1, x2))

@generate.as_vectorized(block_handles=True)
def compute_area_forces(area : Area, detail_nodes):
    x0 = na.node_x(detail_nodes, area.node_IDs[0])
    x1 = na.node_x(detail_nodes, area.node_IDs[1])
    x2 = na.node_x(detail_nodes, area.node_IDs[2])
    forces = area_lib.elastic_area_forces(x0, x1, x2, area.rest_area, area.stiffness, (True, True, True))
    area.f[0] = forces[0]
    area.f[1] = forces[1]
    area.f[2] = forces[2]

@generate.as_vectorized(block_handles=True)
def compute_area_jacobians(area : Area, detail_nodes):
    x0 = na.node_x(detail_nodes, area.node_IDs[0])
    x1 = na.node_x(detail_nodes, area.node_IDs[1])
    x2 = na.node_x(detail_nodes, area.node_IDs[2])
    jacobians = area_lib.elastic_area_numerical_jacobians(x0, x1, x2, area.rest_area, area.stiffness)
    area.dfdx[0][0] = jacobians[0]
    area.dfdx[1][1] = jacobians[1]
    area.dfdx[2][2] = jacobians[2]
    area.dfdx[0][1] = area.dfdx[1][0] = jacobians[3]
    area.dfdx[0][2] = area.dfdx[2][0] = jacobians[4]
    area.dfdx[1][2] = area.dfdx[2][1] = jacobians[5]
