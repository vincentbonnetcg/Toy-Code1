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
        np_block_ids = np.array(block_ids)
        compute_spring_forces(details.area, details.node, np_block_ids)

    @classmethod
    def compute_jacobians(cls, blocks_iterator, details, block_ids=None) -> None:
        np_block_ids = np.array(block_ids)
        compute_spring_jacobians(details.area, details.node, np_block_ids)

@generate.as_vectorized(njit=True, parallel=False, debug=False, block_ids=True)
def compute_area_rest(area : Area, detail_nodes):
    x0 = na.node_x(detail_nodes, area.node_IDs[0])
    x1 = na.node_x(detail_nodes, area.node_IDs[1])
    x2 = na.node_x(detail_nodes, area.node_IDs[2])
    area.rest_area = np.float64(math2D.area(x0, x1, x2))

@generate.as_vectorized(njit=True, parallel=False, debug=False, block_ids=True)
def compute_spring_forces(area : Area, detail_nodes):
    x0 = na.node_x(detail_nodes, area.node_IDs[0])
    x1 = na.node_x(detail_nodes, area.node_IDs[1])
    x2 = na.node_x(detail_nodes, area.node_IDs[2])
    forces = area_lib.elastic_area_forces(x0, x1, x2, area.rest_area, area.stiffness, (True, True, True))
    area.f[0] = forces[0]
    area.f[1] = forces[1]
    area.f[2] = forces[2]

@generate.as_vectorized(njit=True, parallel=False, debug=False, block_ids=True)
def compute_spring_jacobians(area : Area, detail_nodes):
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
