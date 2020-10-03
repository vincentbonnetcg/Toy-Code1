## Those scripts are in the Python SOPs of tube_rig.hipnc

## -------------------------- ##
## Script to generate dataset ##
## -------------------------- ## 
import hou
import os
import sys

working_dir = os.path.dirname(hou.hipFile.path())
if not working_dir in sys.path:
    sys.path.append(working_dir)

import scripts.hou_generate_dataset as gen
reload(gen)

dataset_dir = os.path.dirname(working_dir)

bone_names = ['LowerBack_To_Spine', 'Spine_To_Spine1']
'''
bone_names += ['LeftShoulder_To_LeftArm', 'LeftArm_To_LeftForeArm', 
               'LeftForeArm_To_LeftHand', 'LeftFingerBase_To_LeftHandIndex1']
bone_names += ['RightShoulder_To_RightArm', 'RightArm_To_RightForeArm', 
               'RightForeArm_To_RightHand', 'RightFingerBase_To_RightHandIndex1']
bone_names += ['Neck_To_Neck1', 'Neck1_To_Head','Head_To_HeadEnd']
bone_names += ['LHipJoint_To_LeftUpLeg', 'LeftUpLeg_To_LeftLeg','LeftLeg_To_LeftFoot',
               'LeftFoot_To_LeftToeBase', 'LeftToeBase_To_LeftToeBaseEnd']
bone_names += ['RHipJoint_To_RightUpLeg', 'RightUpLeg_To_RightLeg','RightLeg_To_RightFoot',
               'RightFoot_To_RightToeBase', 'RightToeBase_To_RightToeBaseEnd']
'''
gen.prepare_dataset_dir(dataset_dir)
gen.export_data_from_current_frame(dataset_dir, '/obj/mocapbiped3/', bone_names)


## ---------------------- ##
## Script to read dataset ##
## ---------------------- ##
import hou
import os
import sys

working_dir = os.path.dirname(hou.hipFile.path())
if not working_dir in sys.path:
    sys.path.append(working_dir)

import scripts.hou_evaluate_dataset as ev
reload(ev)

dataset_dir = os.path.dirname(working_dir)
ev.read_dataset_from_current_frame(dataset_dir,  '/obj/mocapbiped3/', prediction=True)

