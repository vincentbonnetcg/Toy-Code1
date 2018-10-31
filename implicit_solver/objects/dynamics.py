"""
@author: Vincent Bonnet
@description : Object descriptions for implicit solver
"""

import constraints as cn
import numpy as np
import objects.shapes as shapes

class Dynamic:
    '''
    Dynamic describes the base class of a dynamic object
    It contains:
    Particle data:
        - num_particles
        - position: x
        - velocity: v
        - mass: m
        - inverse mass: im
        - external forces: f
    Object data / Material:
        - stiffness
        - damping
        - list of internal constraint: internal_constraints[]
    Indexing:
        - global particle offset :globalOffset
        - object index in the scene.dynamics : index
    Render Preferences
        - Render Preferences : renderPrefs
    '''
    def __init__(self, shape, particle_mass, stiffness, damping):
        self.num_particles = shape.num_vertices()
        # Initialize particle state
        self.x = np.zeros((self.num_particles, 2)) # position
        self.v = np.zeros((self.num_particles, 2)) # velocity
        self.m = np.ones(self.num_particles) * particle_mass# mass
        self.im = 1.0 / self.m # inverse mass
        self.f = np.zeros((self.num_particles, 2)) #  force
        # Initialize particle connectivities
        self.edge_ids = np.copy(shape.edge.vertex_ids)
        self.face_ids = np.copy(shape.face.vertex_ids)
        # Useful indices set after adding the object into the scene
        self.global_offset = 0 # global particle offset
        self.index = 0 # object index in the scene.dynamics[.]
        # Material and internal constraints
        self.stiffness = stiffness
        self.damping = damping
        self.internal_constraints = []
        # Copy particle position
        for i in range(self.num_particles):
            self.x[i] = shape.vertex.position[i]
        # Render preferences used by render.py
        # See : https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html for more details
        # fmt = '[color][marker][line]'
        # format of the display State ['particle_fmt', particle_size, 'constraint_fmt', constraint_line_size ]
        self.render_prefs = ['go', 3, 'k-', 1]

    def set_indexing(self, index, global_offset):
        '''
        Sets the global indices (object index and particle offset)
        Those indices are set after the object has been added to the scene
        '''
        self.index = index
        self.global_offset = global_offset

    def create_internal_constraints(self):
        '''
        Creates the internal constraints of the dynamic object.
        It could be used to set materials / distance constraint / area constraint etc.
        '''
        if self.stiffness > 0.0:
            for ids in self.edge_ids:
                self.internal_constraints.append(cn.Spring(self.stiffness, self.damping, [self, self], [ids[0], ids[1]]))

            for ids in self.face_ids:
                self.internal_constraints.append(cn.Area(self.stiffness, self.damping, [self, self, self], [ids[0], ids[1], ids[2]]))


# TODO - remove Wire and use connectivity to create Wire
class Wire(Dynamic):
    '''
    Wire Class describes a dynamic wire object
    '''
    def __init__(self, shape, particle_mass, stiffness, bending_stiffness, damping):
        Dynamic.__init__(self, shape, particle_mass, stiffness, damping)

        self.bending_stiffness = bending_stiffness

    def create_internal_constraints(self):
        for ids in self.edge_ids:
            self.internal_constraints.append(cn.Spring(self.stiffness, self.damping, [self, self], [ids[0], ids[1]]))

        vertex_edges_dict = shapes.vertex_ids_neighbours(self.edge_ids)
        if self.bending_stiffness > 0.0:
            for vertex_id, vertex_id_neighbour in vertex_edges_dict.items():
                if (len(vertex_id_neighbour) == 2):
                    self.internal_constraints.append(cn.Bending(self.bending_stiffness, self.damping, 
                                                                [self, self, self], 
                                                                [vertex_id_neighbour[0], vertex_id, vertex_id_neighbour[1]]))
