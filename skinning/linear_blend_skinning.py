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
        self.num_vertices = len(self.mesh.vertices)
        self.num_bones = len(self.skeleton.bones)
        self.weights = np.zeros((self.num_bones, self.num_vertices))
        self.local_homogenous_vertices = None
        self.local_inv_homogeneous_transforms = None

    def attach_mesh(self, max_influences, kernel_func):
        num_bones = len(self.skeleton.bones)
        bone_segments = self.skeleton.get_bone_segments()

        # Compute weights per bone per vertices (weights map) from kernel function
        for vertex_id, vertex in enumerate(self.mesh.vertices):
            for bone_id, bone_seg in enumerate (bone_segments):
                distance = distance_from_segment(vertex, bone_seg[0], bone_seg[1])
                self.weights[bone_id][vertex_id] = kernel_func(distance)

        # Updates the weights map by limiting ...
        # the number of influences from the n closest vertices
        num_influences = min(num_bones, max_influences)
        for vertex_id, vertex in enumerate(self.mesh.vertices):
            bone_weights = self.weights[:, vertex_id]

            weigths_sorted_index = np.argsort(bone_weights)
            for vtx_id in range(num_bones - num_influences):
                bone_weights[weigths_sorted_index[vtx_id]] = 0.0

            bone_weights /= np.sum(bone_weights)
            self.weights[:, vertex_id] = bone_weights

        # Store the local inverse homogeneous transform and local homogeneous vector
        self.local_inv_homogeneous_transforms = np.zeros((self.num_bones,3,3))
        bone_transforms = self.skeleton.get_bone_homogenous_transforms()
        for bone_id, bone_transform in enumerate(bone_transforms):
            self.local_inv_homogeneous_transforms[bone_id] = np.linalg.inv(bone_transform)

        # create homogenous vertices
        self.local_homogenous_vertices = np.ones((self.num_vertices, 3))
        self.local_homogenous_vertices[:, 0:2] = self.mesh.vertices

        # Update the world space vertices from the local_homogenous_vertex
        # It should not modify the current configuration and only used for debugging
        self.update_mesh()

    def update_mesh(self):
        ws_homogenous_vtx = self.local_to_world(self.local_homogenous_vertices)
        for vertex_id, vertex in enumerate(ws_homogenous_vtx):
            self.mesh.vertices[vertex_id][0:2] = vertex[0:2]

    def local_to_world(self, local_homogenous_vertices):
        '''
        Convert homogenous vertices from local to world space
        '''
        ws_homogenous_vtx = np.zeros((self.num_vertices, 3))

        bone_transforms = self.skeleton.get_bone_homogenous_transforms()
        total_transform = np.zeros((3,3))

        for vertex_id, local_homogenous_vertex in enumerate(local_homogenous_vertices):
            # Compute total transform matrix
            total_transform.fill(0.0)
            for bone_id, bone_transform in enumerate(bone_transforms):
                weight = self.weights[bone_id][vertex_id]
                invT0 = self.local_inv_homogeneous_transforms[bone_id]
                T = (bone_transforms[bone_id])
                total_transform += np.matmul(T * weight, invT0)

            # Transform mesh vertex
            world_homogenous_vertex = np.matmul(total_transform, local_homogenous_vertex)
            ws_homogenous_vtx[vertex_id] = world_homogenous_vertex

        return ws_homogenous_vtx

    def world_to_local(self, ws_homogenous_vtx):
        '''
        Convert homogenous vertices from world to local space
        ws_homogenous_vtx : world space homogenous vertices
        '''
        local_homogenous_vertices = np.zeros((self.num_vertices, 3))

        bone_transforms = self.skeleton.get_bone_homogenous_transforms()
        total_transform = np.zeros((3,3))

        for vertex_id, world_homogenous_vertex in enumerate(ws_homogenous_vtx):
            # Compute total transform matrix
            total_transform.fill(0.0)
            for bone_id, bone_transform in enumerate(bone_transforms):
                weight = self.weights[bone_id][vertex_id]
                invT0 = self.local_inv_homogeneous_transforms[bone_id]
                T = (bone_transforms[bone_id])
                total_transform += np.matmul(T * weight, invT0)

            # Transform mesh vertex
            inv_total_transform = np.linalg.inv(bone_transform)
            local_homogenous_vertex = np.matmul(inv_total_transform, world_homogenous_vertex)
            local_homogenous_vertices[vertex_id] = local_homogenous_vertex

        return local_homogenous_vertices

