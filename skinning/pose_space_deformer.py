'''
@author: Vincent Bonnet
@description : Pose Space Deformer
'''

'''
PoseSpaceDeformer decouples animation control from target geometry
It maps features with PSD target
a feature can be the angle between two bones
a PSD target is the final geometry or the displacement relative to an existing underlying skinning
This version of PoseSpaceDeformer compute the displacement on top of underlying skinning
'''

def feature_from_skeleton(skeleton):
    return skeleton.get_relative_rotations()

class PoseSpaceDeformer:
    def __init__(self):
        self.features = []
        self.psd_targets = []

    def add_pose(self, feature, psd_target):
        self.features.append(feature)
        self.psd_targets.append(psd_target)


