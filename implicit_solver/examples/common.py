"""
@author: Vincent Bonnet
@description : common functions for scenes
"""

import os

def get_resources_folder():
    return os.path.join(os.path.dirname(__file__), "resources")

def meta_data_render(width=1.0, color='grey', style='solid', alpha = 1.0):
    return {'width':width, 'color':color, 'style':style, 'alpha' : alpha}

