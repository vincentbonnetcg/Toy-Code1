## Those scripts are in the Python SOPs of tube_rig.hipnc
## Script to generate dataset
import hou
import os
import sys

working_dir = os.path.dirname(hou.hipFile.path())
if not working_dir in sys.path:
    sys.path.append(working_dir)

import scripts.hou_generate_dataset as gen
reload(gen)

gen.prepare_dataset_dir(working_dir)
gen.export_data_from_current_frame(working_dir)


## Script to read dataset
import os
import sys

working_dir = os.path.dirname(hou.hipFile.path())
if not working_dir in sys.path:
    sys.path.append(working_dir)

import scripts.hou_evaluate_dataset as ev
reload(ev)

ev.read_dataset_from_current_frame(working_dir)

