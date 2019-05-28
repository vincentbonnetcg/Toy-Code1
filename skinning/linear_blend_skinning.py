"""
@author: Vincent Bonnet
@description : Linear blend skinning algorithm
"""

import numpy as np

def distance_from_segment(p, seg_p1, seg_p2):
    '''
    Distance from segment [seg_p1, seg_p2] and point p
    '''
    d = seg_p2 - seg_p1
    d_norm = np.linalg.norm(d)
    d_normalized = d / d_norm
    t = np.dot(p - seg_p1, d_normalized)
    t = min(max(t, 0.0), d_norm)
    projected_p = seg_p1 + d_normalized * t
    return np.linalg.norm(p - projected_p)

class LinearBlendSkinning:

    def __init__(self, mesh, skeleton):
        self.mesh = mesh
        self.skeleton = skeleton
        num_vertices = len(self.mesh.vertex_buffer)
        num_bones = len(self.skeleton.bones)
        self.weights_map = np.zeros((num_bones, num_vertices))
        self.local_homogenous_vertices = np.zeros((num_vertices, 3))
        self.local_inv_homogeneous_transforms = np.zeros((num_bones,3,3))

    def attach_mesh(self, max_influences, kernel_func):
        num_bones = len(self.skeleton.bones)
        bone_segments = self.skeleton.get_bone_segments()

        # Compute weights per bone per vertices (weights map) from kernel function
        for vertex_id, vertex in enumerate(self.mesh.vertex_buffer):
            for bone_id, bone_seg in enumerate (bone_segments):
                distance = distance_from_segment(vertex, bone_seg[0], bone_seg[1])
                self.weights_map[bone_id][vertex_id] = kernel_func(distance)

        # Updates the weights map by limiting ...
        # the number of influences from the n closest vertices
        num_influences = min(num_bones, max_influences)
        for vertex_id, vertex in enumerate(self.mesh.vertex_buffer):
            vertex_weights = np.zeros(num_bones)
            for bone_id, bone_seg in enumerate (bone_segments):
                vertex_weights[bone_id] = self.weights_map[bone_id][vertex_id]

            vertex_weigths_sorted_index = np.argsort(vertex_weights)
            for vtx_id in range(num_bones - num_influences):
                vertex_weights[vertex_weigths_sorted_index[vtx_id]] = 0.0

            vertex_weights /= np.sum(vertex_weights)
            for bone_id, bone_seg in enumerate (bone_segments):
                self.weights_map[bone_id][vertex_id] = vertex_weights[bone_id]

        # Store the local inverse homogeneous transform and local homogeneous vector
        bone_transforms = self.skeleton.get_bone_homogenous_transforms()
        for bone_id, bone_transform in enumerate(bone_transforms):
            self.local_inv_homogeneous_transforms[bone_id] = np.linalg.inv(bone_transform)

        for vertex_id, vertex in enumerate(self.mesh.vertex_buffer):
            local_vertex_homogenous = np.ones(3)
            local_vertex_homogenous[0:2] = vertex
            self.local_homogenous_vertices[vertex_id] = local_vertex_homogenous

        # Update the world space vertices from the local_homogenous_vertex
        # It should not modify the current configuration and only used for debugging
        self.update_mesh()

    def update_mesh(self):
        world_homogenous_vertices = self.local_to_world(self.local_homogenous_vertices)
        for vertex_id, vertex in enumerate(world_homogenous_vertices):
            self.mesh.vertex_buffer[vertex_id][0:2] = vertex[0:2]

    def local_to_world(self, local_homogenous_vertices):
        '''
        Convert homogenous vertices from local to world space
        local_homogenous_vertices should be the size of the self.mesh.vertex_buffer
        '''
        assert(len(local_homogenous_vertices) == len(self.local_homogenous_vertices))
        world_homogenous_vertices = np.zeros((len(local_homogenous_vertices), 3))

        bone_transforms = self.skeleton.get_bone_homogenous_transforms()
        for vertex_id, local_homogenous_vertex in enumerate(local_homogenous_vertices):
            # Compute total transform matrix
            total_transform = np.zeros((3,3))
            for bone_id, bone_transform in enumerate(bone_transforms):
                weight = self.weights_map[bone_id][vertex_id]
                invT0 = self.local_inv_homogeneous_transforms[bone_id]
                T = (bone_transforms[bone_id])
                total_transform += np.matmul(T * weight, invT0)

            # Transform mesh vertex
            world_homogenous_vertex = np.matmul(total_transform, local_homogenous_vertex)
            world_homogenous_vertices[vertex_id] = world_homogenous_vertex

        return world_homogenous_vertices

    def world_to_local(self, world_homogenous_vertices):
        '''
        Convert homogenous vertices from world to local space
        world_homogenous_vertices should be the size of the self.mesh.vertex_buffer
        '''
        assert(len(world_homogenous_vertices) == len(self.local_homogenous_vertices))
        local_homogenous_vertices = np.zeros((len(world_homogenous_vertices), 3))

        bone_transforms = self.skeleton.get_bone_homogenous_transforms()
        for vertex_id, world_homogenous_vertex in enumerate(world_homogenous_vertices):
            # Compute total transform matrix
            total_transform = np.zeros((3,3))
            for bone_id, bone_transform in enumerate(bone_transforms):
                weight = self.weights_map[bone_id][vertex_id]
                invT0 = self.local_inv_homogeneous_transforms[bone_id]
                T = (bone_transforms[bone_id])
                total_transform += np.matmul(T * weight, invT0)

            # Transform mesh vertex
            inv_total_transform = np.linalg.inv(bone_transform)
            local_homogenous_vertex = np.matmul(inv_total_transform, world_homogenous_vertex)
            local_homogenous_vertices[vertex_id] = local_homogenous_vertex

        return local_homogenous_vertices
