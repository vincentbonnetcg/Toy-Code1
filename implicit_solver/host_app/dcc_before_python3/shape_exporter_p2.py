"""
@author: Vincent Bonnet
@description : This file contains serializer and deserializer of shapes
It is only uses because Houdini and Maya do not support Python3 yet.
This file will go away as soon as those two DCCs move to Python3
"""

import pickle

def write_to_file(points, edge_ids, face_ids, filename):
    out_file = open(filename,'wb')
    out_dict = {'points' : points, 'edge_ids' : edge_ids, 'face_ids' : face_ids }
    pickle.dump(out_dict, out_file)
    out_file.close()

def read_from_file(filename):
    in_file = open(filename,'rb')
    in_dict = pickle.load(in_file)
    in_file.close()
    points = in_dict['points']
    edge_ids = in_dict['edge_ids']
    face_ids = in_dict['face_ids']
    return points, edge_ids, face_ids
