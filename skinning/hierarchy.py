"""
@author: Vincent Bonnet
@description : Skeleton and Bone class
"""

# TODO - see bone_children[0] : add support for multiple children per bone

import numpy as np

def create_skeleton_with_4_bones():
    '''
    Create a skeleton object
    '''
    root_bone = Bone(length = 3.0, rotation = 0.0)
    bone1 = Bone(length = 3.0, rotation = 0.0)
    bone2 = Bone(length = 3.0, rotation = 0.0)
    bone3 = Bone(length = 3.0, rotation = 0.0)

    root_bone.rotation_animation = lambda time : np.sin(time / 2.0 * np.pi) * 20.0
    bone1.rotation_animation = lambda time : np.sin(time / 2.0 * np.pi) * 24.0
    bone2.rotation_animation = lambda time : np.sin(time / 2.0 * np.pi) * 32.0
    bone3.rotation_animation = lambda time : np.sin(time / 2.0 * np.pi) * 36.0

    skeleton = Skeleton([-6.0, 0.0], root_bone)
    skeleton.add_bone(root_bone)
    skeleton.add_bone(bone1, root_bone)
    skeleton.add_bone(bone2, bone1)
    skeleton.add_bone(bone3, bone2)

    return skeleton

def create_skeleton_with_2_bones():
    '''
    Create a skeleton object
    '''
    root_bone = Bone(length = 6.0, rotation = 0.0)
    bone1 = Bone(length = 6.0, rotation = 0.0)

    root_bone.rotation_animation = lambda time : np.sin(time / 2.0 * np.pi) * 0
    bone1.rotation_animation = lambda time : np.sin(time / 2.0 * np.pi) * 90

    skeleton = Skeleton([-6.0, 0.0], root_bone)
    skeleton.add_bone(root_bone)
    skeleton.add_bone(bone1, root_bone)

    return skeleton


class Bone:
    def __init__(self, length = 1.0, rotation = 0.0):
        self.length = length
        self.rotation = rotation # in degrees
        self.rotation_animation = lambda time : rotation
        # hirerarchy info
        self.bone_parent = None
        self.bone_children = []

    def get_homogenous_transform(self):
        '''
        3x3 Matrix combining rotation and displacement
        where R is 2x2 rotation matrix
        and d is 2d vector
        | R  d |
        | 0  1 |
        '''
        H = np.zeros((3,3))
        cos = np.cos(np.deg2rad(self.rotation))
        sin = np.sin(np.deg2rad(self.rotation))
        H[0, 0] = cos
        H[1, 1] = cos
        H[0, 1] = -sin
        H[1, 0] = sin
        H[0, 2] = cos * self.length
        H[1, 2] = sin * self.length
        H[2, 2] = 1.0
        return H

    def animate(self, time):
        self.rotation = self.rotation_animation(time)

class Skeleton:
    def __init__(self, root_position, root_bone):
        self.root_position = np.asarray(root_position)
        self.root_bone = root_bone
        self.bones = []

    def add_bone(self, bone, bone_parent = None):
        self.bones.append(bone)
        if bone_parent is not None:
            bone.bone_parent = bone_parent
            bone_parent.bone_children.append(bone)

    def get_homogenous_transform(self):
        H = np.identity(3)
        H[0, 2] = self.root_position[0]
        H[1, 2] = self.root_position[1]
        return H

    def get_relative_rotations(self):
        num_bones = len(self.bones)
        relative_rotations = np.zeros(num_bones)

        bone_id = 0

        bone = self.root_bone
        while bone is not None:

            relative_rotation = 0.0
            if bone.bone_parent:
                relative_rotation = bone.rotation - bone.bone_parent.rotation

            # Go to the children
            if len(bone.bone_children) > 0:
                bone = bone.bone_children[0]
            else:
                bone = None

            relative_rotations[bone_id] = relative_rotation
            bone_id += 1

        return relative_rotations


    def get_bone_homogenous_transforms(self):
        '''
        Returns the world space transform of each bones
        '''
        num_bones = len(self.bones)
        bone_transforms = np.zeros((num_bones,3,3))

        H = self.get_homogenous_transform()
        bone_id = 0

        bone = self.root_bone
        while bone is not None:
            # Concatenate transformation matrice
            bone_H = bone.get_homogenous_transform()
            H = np.matmul(H, bone_H)

            # Go to the children
            if len(bone.bone_children) > 0:
                bone = bone.bone_children[0]
            else:
                bone = None

            bone_transforms[bone_id] = H
            bone_id += 1

        return bone_transforms

    def get_bone_segments(self):
        homogenous_coordinate = np.asarray([0.0, 0.0, 1.0])
        bone_transforms = self.get_bone_homogenous_transforms()

        segments = []

        H = self.get_homogenous_transform()

        prev_pos = np.matmul(H, homogenous_coordinate)

        for bone_id, bone_H in enumerate(bone_transforms):

            next_pos = np.matmul(bone_H, homogenous_coordinate)
            segments.append([prev_pos[0:2], next_pos[0:2]])
            prev_pos = next_pos

        return segments

    def animate(self, time):
        for bone in self.bones:
            bone.animate(time)
