"""
@author: Vincent Bonnet
@description : Python code to bridge Maya Data to the solver
This code should be run from Maya Python Script Editor
"""
import maya.api.OpenMaya as om
import pickle

FILENAME = "" # ADD FILE

def extract_tri_mesh_data(points, edge_ids, face_ids):
    # get selected mesh
    selection = om.MSelectionList()
    selection = om.MGlobal.getActiveSelectionList()
    iter_sel = om.MItSelectionList(selection, om.MFn.kMesh)

    if iter_sel.isDone():
        print("no mesh selected")
        return

    dag_path = iter_sel.getDagPath()
    fn_mesh = om.MFnMesh(dag_path)

    # get vertices
    pts = fn_mesh.getPoints(om.MSpace.kWorld)
    for i in range(len(pts)) :
        points.append((pts[i].x, pts[i].z))

    # get edge indices
    edge_iter = om.MItMeshEdge(dag_path)
    while not edge_iter.isDone():
        v0 = edge_iter.vertexId(0)
        v1 = edge_iter.vertexId(1)
        edge_ids.append((v0, v1))
        edge_iter.next()

    # get faces
    polygon_iter = om.MItMeshPolygon(dag_path)
    while not polygon_iter.isDone():
        v0 = polygon_iter.vertexIndex(0)
        v1 = polygon_iter.vertexIndex(1)
        v2 = polygon_iter.vertexIndex(2)
        face_ids.append((v0, v1, v2))
        polygon_iter.next(0)

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

points = []
edge_ids = []
face_ids = []
extract_tri_mesh_data(points, edge_ids, face_ids)

write_to_file(points, edge_ids, face_ids, FILENAME)

points, edge_ids, face_ids = read_from_file(FILENAME)
print(points)
print(edge_ids)
print(face_ids)

