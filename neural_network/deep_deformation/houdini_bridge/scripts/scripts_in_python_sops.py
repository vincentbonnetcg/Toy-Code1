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
gen.prepare_dataset_dir(dataset_dir)
gen.export_data_from_current_frame(dataset_dir, '/obj/mocapbiped3/')


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

